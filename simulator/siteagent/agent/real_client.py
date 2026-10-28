"""Guarded client for real Token Factory calls. Not used by the offline demo, which calls no model.

What it enforces before and around a send:
- the SDK client is created with max_retries=0 (no hidden retry) and a finite timeout; the credential comes from the
  mapping it is given (default: the process environment), is never passed to the program's child process and never
  printed or logged;
- preflight(): the model catalog entry is checked with tf_guardrails.validate_pricing (reviewed model id, context
  length, token prices, no unreviewed charge) before any completion request;
- complete(): refuses a request that is not the reviewed shape (model, messages with exactly role and content,
  temperature, max_completion_tokens within the limit, optional reasoning_effort equal to the run's declared one);
  each send is bound to one ledger reservation, consumed before the transport, and makes exactly one transport call;
  a technical failure raises TransientLLMError (the loop decides about its single shared retry), anything else
  raises LLMRefused; exception texts are never kept, only the exception type name;
- an operational log (JSONL) keeps time, model id, request fingerprint, status, finish reason, usage and latency, never
  headers, keys or message contents; redact() removes credential-like keys and values as a second barrier;
- claim_run_authorization() / authorize_run(): a one-shot marker per approved run, and an envelope (max calls, max
  cost) enforced by the bound ledger and by the client itself.
Financial and attempt ceilings may be None in the active configuration; reservations, accounting, timeouts, identity,
price and request checks still apply.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .ledger import MAX_COMPLETION_TOKENS, RESERVED_PROMPT_TOKENS, BudgetStop, _entry_cost, model_pricing
from .llm import (MODEL_ID, REASONING_EFFORTS, ClientNotReady, LLMRefused, LLMResponse, TransientLLMError,
                  request_fingerprint, valid_reasoning_effort)

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"  # tf_first_call.BASE_URL (test checks they match)
CREDENTIAL_ENV = "NEBIUS_API_KEY"  # tf_first_call reads the same variable from its environment
REQUEST_KEYS = frozenset({"model", "messages", "temperature", "max_completion_tokens"})  # required, exactly
OPTIONAL_REQUEST_KEYS = frozenset({"reasoning_effort"})  # only with a documented value (llm.REASONING_EFFORTS)
REAL_TRANSPORT_KIND = "token_factory"  # set by openai_transport; any other kind is labelled as not real
DEFAULT_TIMEOUT_S = 120.0
_SECRET_KEY = re.compile(r"(authori[sz]ation|api[_-]?key|header|cookie|token$|secret|password|credential)", re.I)
# left boundary: without it "task-criteria-1.4" became "ta[redacted].4"
_SECRET_VAL = re.compile(r"(Bearer\s+\S+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{8,}|(?<![A-Za-z0-9])v1\.[A-Za-z0-9_\-]{20,})")
FAKE_TRANSPORT_KIND = "fake_dry_run"  # scripts/run_agent_real.FakeTransport (dry run, no network)


class TransportTransient(Exception):
    """Raised by a transport for a technical failure that may be retried once per run."""


class TransportError(Exception):
    """Raised by a transport for any other failure (not retried)."""


class LLMAnsweredRefused(LLMRefused):
    """An ANSWER came back (so it was billed) but cannot be used: empty content, or another model named.
    Carries what the provider reported (usage, finish_reason, model id), never the text, so the loop can settle on
    provider usage and label the case (before, an empty answer became 'model unavailable' with no usage and the full
    reservation kept as uncertain)."""

    def __init__(self, code: str, usage, finish_reason, model_id, latency_s: float, usage_source: str,
                 reasoning_chars: int = 0):
        super().__init__(code)
        self.code, self.usage, self.finish_reason = code, usage, finish_reason
        self.model_id, self.latency_s, self.usage_source = model_id, latency_s, usage_source
        self.reasoning_chars = reasoning_chars


def usage_source_for(transport_kind: str) -> str:
    """'provider' only for the real Token Factory transport."""
    if transport_kind == REAL_TRANSPORT_KIND:
        return "provider"
    return "fake_transport" if transport_kind == FAKE_TRANSPORT_KIND else "unknown_transport"


KNOWN_TRANSIENT = ("APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError")
# non-transient SDK error names a transport may report (openai_transport passes type(e).__name__ only). Logged from this
# fixed list only, never free text: BadRequestError (HTTP 400) is how an unsupported option would show.
KNOWN_REFUSALS = ("BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError", "ConflictError",
                  "UnprocessableEntityError", "APIStatusError", "APIResponseValidationError", "APIError")


def reasoning_tokens_reported(raw_usage) -> int | None:
    """usage.completion_tokens_details.reasoning_tokens when the provider reports it as a non-negative int (an
    optional extra, recorded, never used for cost); None otherwise (absent, null, malformed)."""
    details = raw_usage.get("completion_tokens_details") if isinstance(raw_usage, dict) else None
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return value if type(value) is int and value >= 0 else None


def _transient_name(e) -> str:
    """The SDK error type a transport reports, if it is a known name; else the class name only."""
    t = str(e)
    return t if t in KNOWN_TRANSIENT else type(e).__name__


def redact(obj, depth: int = 0):
    """Copy of obj without credential-like keys or values (second barrier for logs)."""
    if depth > 12:
        return "[too deep]"
    if isinstance(obj, dict):
        return {k: ("[redacted]" if _SECRET_KEY.search(str(k)) and k not in ("total_tokens", "prompt_tokens",
                                                                           "completion_tokens", "max_completion_tokens")
                    else redact(v, depth + 1)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(x, depth + 1) for x in obj]
    if isinstance(obj, str):
        return _SECRET_VAL.sub("[redacted]", obj)
    return obj


def finite_timeout(timeout_s) -> float:
    """No financial cap does not disable the transport's finite positive time bound."""
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError, OverflowError):
        raise ClientNotReady("timeout_s must be finite and positive") from None
    if isinstance(timeout_s, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ClientNotReady("timeout_s must be finite and positive")
    return timeout


def openai_transport(timeout_s: float = DEFAULT_TIMEOUT_S, environ=None, openai_module=None):
    """Transport over the OpenAI SDK (Token Factory is OpenAI-compatible): max_retries=0, timeout.
    The credential comes from `environ` (default: this process's environment), never a file.
    openai_module: injected by the tests (a fake); real use imports the installed SDK."""
    timeout_s = finite_timeout(timeout_s)
    env = os.environ if environ is None else environ
    key = env.get(CREDENTIAL_ENV)
    if not key:
        raise ClientNotReady("credential not present in the process environment; no request made")
    if openai_module is None:  # pragma: no cover (real SDK, not used in this track)
        import openai as openai_module
    client = openai_module.OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=timeout_s)
    del key
    transient = tuple(getattr(openai_module, n) for n in ("APITimeoutError", "APIConnectionError", "RateLimitError",
                                                         "InternalServerError") if hasattr(openai_module, n))
    base = getattr(openai_module, "OpenAIError", Exception)

    def send(request: dict, timeout: float) -> dict:
        try:
            resp = client.chat.completions.create(**request, timeout=timeout)
        except transient as e:
            raise TransportTransient(type(e).__name__) from None
        except base as e:
            raise TransportError(type(e).__name__) from None
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    def models() -> list[dict]:
        """The live catalog WITH prices: GET /models?verbose=true, as tf_first_call did (the plain
        list has no pricing field, so the pricing check could not run on it)."""
        try:
            catalog = client.get("/models", cast_to=dict, options={"params": {"verbose": "true"}})
        except base as e:
            raise TransportError(type(e).__name__) from None
        data = catalog.get("data") if isinstance(catalog, dict) else None
        if not isinstance(data, list):
            raise TransportError("InvalidCatalog")
        return [m for m in data if isinstance(m, dict)]

    send.models = models
    send.kind = REAL_TRANSPORT_KIND
    return send


class GuardedRealClient:
    """complete(request, reservation) -> LLMResponse, for agent.loop.AgentLoop (binds_reservations: the
    loop passes the id of the reservation it made for this send)."""

    binds_reservations = True
    client_kind = "real"

    def __init__(self, transport, ledger, timeout_s: float = DEFAULT_TIMEOUT_S, log_path: str | Path | None = None,
                 model: str = MODEL_ID, authorization: "RunAuthorization | None" = None):
        """authorization: from authorize_run(); ledger must be authorization.bind(<run ledger>)."""
        self._send, self._ledger, self._timeout = transport, ledger, finite_timeout(timeout_s)
        self._auth = authorization
        self._log = None if log_path is None else Path(log_path)
        try:
            model_pricing(model)
        except ValueError:
            raise ClientNotReady("the client's model is not a reviewed model; no request made") from None
        self.model_id = model
        self.request_model = model  # agent.loop builds its requests for this model
        self.transport_kind = getattr(transport, "kind", "unknown")
        self.pricing = None
        self.n = 0
        self.received: list[dict] = []  # the requests sent (copies), like MockLLMClient (no secret inside)

    def preflight(self, catalog: list[dict] | None = None):
        """Catalog and pricing check before any completion (tf_guardrails.validate_pricing)."""
        from tf_guardrails import GuardrailError, validate_pricing
        if catalog is None:
            catalog = self._send.models()
        same = [m for m in catalog if isinstance(m, dict) and m.get("id") == self.model_id]
        if len(same) > 1:
            raise ClientNotReady("catalog or pricing check failed: the model appears more than once")
        entry = same[0] if same else None
        try:
            self.pricing = validate_pricing(entry, self.model_id)
        except GuardrailError as e:
            raise ClientNotReady(f"catalog or pricing check failed: {e}") from None
        return self.pricing

    def _check_request(self, request: dict):
        if not isinstance(request, dict) or not REQUEST_KEYS <= set(request) \
                or set(request) - REQUEST_KEYS - OPTIONAL_REQUEST_KEYS:
            raise ClientNotReady(f"request keys must be exactly {sorted(REQUEST_KEYS)}, plus optionally "
                                 f"{sorted(OPTIONAL_REQUEST_KEYS)}")
        if "reasoning_effort" in request and not valid_reasoning_effort(request["reasoning_effort"]):
            raise ClientNotReady(f"reasoning_effort must be one of {list(REASONING_EFFORTS)} (or absent); "
                                 "no request made")
        if request["model"] != self.model_id:
            raise ClientNotReady("request for another model than the reviewed one")
        mct = request["max_completion_tokens"]
        if type(mct) is not int or not 1 <= mct <= MAX_COMPLETION_TOKENS:
            raise ClientNotReady(f"max_completion_tokens must be an int in [1, {MAX_COMPLETION_TOKENS}]")
        if not isinstance(request["messages"], list) or not request["messages"]:
            raise ClientNotReady("messages missing")
        from tf_guardrails import GuardrailError, prompt_token_bound
        try:
            bound = prompt_token_bound(request["messages"])  # closed message schema (role and content only)
        except GuardrailError as e:
            raise ClientNotReady(f"messages refused ({e}); no request made") from None
        if bound > RESERVED_PROMPT_TOKENS:
            raise ClientNotReady(f"prompt bound {bound} tokens > reserved {RESERVED_PROMPT_TOKENS}; no request made")

    def _write_log(self, row: dict):
        if self._log is None:
            return
        self._log.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log, "a", encoding="utf-8") as f:
            f.write(json.dumps(redact(row), sort_keys=True) + "\n")

    def complete(self, request: dict, reservation=None) -> LLMResponse:
        if self.pricing is None:
            raise ClientNotReady("preflight (catalog and pricing check) not done; no request made")
        if self._auth is None or not isinstance(self._ledger, EnvelopeLedger) or self._ledger.authorization is not self._auth:
            raise ClientNotReady("no run authorization bound to this client and its ledger; no request made")
        if self._auth.max_calls is not None and self.n >= self._auth.max_calls:
            raise ClientNotReady(f"the run envelope allows {self._auth.max_calls} calls; no request made")
        self._check_request(request)
        declared = self._auth.envelope.get("reasoning_effort")  # the run's declared setting (None: not sent)
        if request.get("reasoning_effort") != declared:
            raise ClientNotReady(f"request reasoning_effort {request.get('reasoning_effort')!r} differs from the declared "
                                 f"envelope ({declared!r}); no request made")
        if reservation is None:
            raise ClientNotReady("each send needs the id of its own cost reservation (agent.ledger); no request made")
        self._ledger.consume_for_send(reservation)  # one send per reservation, consumed BEFORE the transport
        self.n += 1
        self.received.append(json.loads(json.dumps(request)))
        fp = request_fingerprint(request)
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model_id, "request_sha256": fp, "n": self.n,
               "reservation_id": reservation, "transport": self.transport_kind,
               "max_completion_tokens": request["max_completion_tokens"],
               "reasoning_effort": request.get("reasoning_effort")}  # what was sent (None: no such key)
        t0 = time.monotonic()
        try:
            raw = self._send(request, self._timeout)
        except TransportTransient as e:
            name = _transient_name(e)  # a known SDK type name only, never the transport's free text
            self._write_log({**row, "status": "transient_error", "error_type": name,
                             "latency_s": round(time.monotonic() - t0, 3)})
            raise TransientLLMError(f"transient transport failure ({name})") from None
        except Exception as e:  # noqa: BLE001 (type name only, never the text)
            sdk = str(e) if isinstance(e, TransportError) and str(e) in KNOWN_REFUSALS else None  # fixed names only
            self._write_log({**row, "status": "error", "error_type": type(e).__name__, "sdk_error": sdk,
                             "latency_s": round(time.monotonic() - t0, 3)})
            raise LLMRefused(f"request failed ({type(e).__name__}{': ' + sdk if sdk else ''})") from None
        latency = round(time.monotonic() - t0, 3)
        try:
            ch = raw["choices"][0]
            msg = ch["message"]
            text = msg["content"]
            finish = ch.get("finish_reason")
            usage = {k: raw["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            got_model = raw.get("model")
        except (KeyError, IndexError, TypeError):
            self._write_log({**row, "status": "malformed_response", "latency_s": latency})
            raise LLMRefused("malformed response") from None
        # a separate reasoning field (if the provider returns one) is kept bench side, never executed
        field_name = next((k for k in ("reasoning_content", "reasoning")
                           if isinstance(msg, dict) and isinstance(msg.get(k), str)), None)
        reasoning = msg[field_name] if field_name else None
        # Optional extra of the provider, recorded with the usage, never used for cost (the three fields above are)
        rt = reasoning_tokens_reported(raw.get("usage"))
        if rt is not None:
            usage["completion_tokens_details"] = {"reasoning_tokens": rt}
        row.update(reasoning_field=field_name, reasoning_tokens=rt)
        source = usage_source_for(self.transport_kind)
        if got_model != self.model_id:
            self._write_log({**row, "status": "unexpected_model", "latency_s": latency, "usage": usage,
                             "finish_reason": finish})
            raise LLMAnsweredRefused("unexpected_model", usage, finish, str(got_model)[:80], latency, source,
                                     len(reasoning or ""))
        if not isinstance(text, str) or not text.strip():
            self._write_log({**row, "status": "empty", "latency_s": latency, "usage": usage, "finish_reason": finish,
                             "reasoning_chars": len(reasoning or "")})
            raise LLMAnsweredRefused("empty_answer", usage, finish, got_model, latency, source, len(reasoning or ""))
        self._write_log({**row, "status": "answered", "latency_s": latency, "usage": usage, "finish_reason": finish,
                         # A lone surrogate in a paid answer no longer raises here (the answer was lost)
                         "answer_sha256": hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest(),
                         "reasoning_chars": len(reasoning or "")})
        return LLMResponse(text=text, finish_reason=str(finish), usage=usage, model_id=got_model,
                           latency_s=latency, usage_source=source, reasoning=reasoning)


def claim_run_authorization(marker_dir: str | Path, authorization_id: str, envelope: dict) -> Path:
    """One-shot marker of an approved run (like the first-call script): created exclusively, never erased. A second
    claim of the same id is refused. envelope: model, prompt version, limits version, maximum calls
    and maximum cost of the run, as approved by the operator (no secret)."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]{6,80}", authorization_id or ""):
        raise ClientNotReady("authorization id: 6 to 80 letters, digits, _ or -")
    need = {"model", "prompt_version", "limits_version", "max_calls", "max_cost_usd"}
    if not need <= set(envelope):
        raise ClientNotReady(f"envelope needs {sorted(need)}")
    p = Path(marker_dir) / f"{authorization_id}.used.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise ClientNotReady("this authorization was already used; a new run needs a new approval") from None
    with os.fdopen(fd, "w") as f:
        json.dump({"authorization_id": authorization_id, "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "envelope": redact(envelope)}, f, indent=1)
    return p


@dataclass
class RunAuthorization:
    """One approved real run: its one-shot marker and its envelope (max_calls, max_cost_usd)."""
    authorization_id: str
    envelope: dict
    marker: Path
    max_calls: int | None
    max_cost_usd: Decimal | None
    bound: list = field(default_factory=list)
    deadline_monotonic: float | None = None  # time.monotonic() after which no reservation is made (run_real)

    def bind(self, ledger) -> "EnvelopeLedger":
        """The run's ledger, wrapped so that no reservation goes beyond the envelope. One per run."""
        if self.bound:
            raise ClientNotReady("this authorization is already bound to a ledger (one ledger per run)")
        e = EnvelopeLedger(ledger, self)
        self.bound.append(e)
        return e


def envelope_caps(envelope: dict) -> tuple[int | None, Decimal | None]:
    """Explicit JSON null means no cap. Missing, non-finite and malformed values are refused."""
    try:
        max_calls = envelope["max_calls"]
        raw_cost = envelope["max_cost_usd"]
        if max_calls is not None and (type(max_calls) is not int or max_calls < 1):
            raise ValueError
        if isinstance(raw_cost, bool):
            raise ValueError
        max_cost = None if raw_cost is None else Decimal(str(raw_cost))
        if max_cost is not None and (not max_cost.is_finite() or max_cost <= 0):
            raise ValueError
    except (KeyError, TypeError, ValueError, InvalidOperation):
        raise ClientNotReady("envelope needs positive integer max_calls and finite positive max_cost_usd, or explicit null") from None
    return max_calls, max_cost


def money_text(value) -> str | None:
    """JSON-safe amount: absence of a cap stays null, never the string 'None' or Infinity."""
    return None if value is None else str(value)


def authorize_run(marker_dir: str | Path, authorization_id: str, envelope: dict) -> RunAuthorization:
    """Claim the approved run ONCE; explicit null leaves the corresponding envelope cap unset."""
    max_calls, max_cost = envelope_caps(envelope)
    try:
        res = model_pricing(envelope.get("model")).reservation_usd
    except ValueError:
        raise ClientNotReady("envelope model is not a reviewed model") from None
    if max_cost is not None and max_cost < res:
        raise ClientNotReady(f"envelope too small: max_calls >= 1 and max_cost_usd >= one reservation ({res})")
    p = claim_run_authorization(marker_dir, authorization_id, envelope)
    return RunAuthorization(authorization_id, dict(envelope), p, max_calls, max_cost)


class EnvelopeLedger:
    """The run's ledger (MemoryLedger or GuardedLedger) behind the approved envelope: reserve() is
    refused when the next reservation would go beyond max_calls or max_cost_usd (committed cost =
    settled costs + uncertain amounts + open reservations of this run). Everything else is passed
    through. The underlying ledger keeps its own caps."""

    def __init__(self, ledger, authorization: RunAuthorization):
        self._l, self.authorization = ledger, authorization
        self.reserved_calls = 0

    @property
    def entries(self):
        return self._l.entries

    @property
    def reservation_usd(self):
        """The per-call reservation of the underlying ledger (its model's bounded worst case)."""
        return getattr(self._l, "reservation_usd", model_pricing(None).reservation_usd)

    def committed_usd(self) -> Decimal:
        return sum((_entry_cost(e, Decimal(e["reserved_usd"])) for e in self._l.entries), Decimal(0))

    def would_refuse(self) -> str | None:
        """Why the next reservation would be refused by the envelope (None: allowed). The loop asks this
        before spending the run's technical retry."""
        a = self.authorization
        if a.max_calls is not None and self.reserved_calls + 1 > a.max_calls:
            return f"run envelope: max_calls ({a.max_calls}) reached"
        if a.max_cost_usd is not None and self.committed_usd() + self.reservation_usd > a.max_cost_usd:
            return f"run envelope: max_cost_usd ({a.max_cost_usd}) would be exceeded by the next reservation"
        if a.deadline_monotonic is not None and time.monotonic() > a.deadline_monotonic:
            return "run envelope: stage wall-time deadline reached"
        return None

    def reserve(self, activity: str):
        why = self.would_refuse()
        if why is not None:
            raise BudgetStop(why)
        tok = self._l.reserve(activity)
        self.reserved_calls += 1
        return tok

    def consume_for_send(self, token) -> None:
        self._l.consume_for_send(token)

    def settle(self, token, response, prompt_version):
        return self._l.settle(token, response, prompt_version)

    def keep_uncertain(self, token, reason):
        return self._l.keep_uncertain(token, reason)

    def has_open_reservation(self) -> bool:
        return self._l.has_open_reservation()

    def summary(self) -> dict:
        base = self._l.summary() if hasattr(self._l, "summary") else {}
        return {**base, "envelope": {"authorization_id": self.authorization.authorization_id,
                                     "max_calls": self.authorization.max_calls,
                                     "max_cost_usd": money_text(self.authorization.max_cost_usd),
                                     "reserved_calls": self.reserved_calls, "committed_usd": str(self.committed_usd())}}
