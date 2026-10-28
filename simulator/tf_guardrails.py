"""Local cost checks for Token Factory calls. This module never reads credentials.

It holds the reviewed models and their frozen prices (REVIEWED_MODELS, checked against the live catalog at launch by
validate_pricing), the request bound (prompt_token_bound: UTF-8 bytes of every message plus a fixed overhead must stay
under RESERVED_PROMPT_TOKENS, with a closed message schema), and the budget configuration (load_budget_config).

A reservation is a bounded worst case, the same rule for every model: RESERVED_PROMPT_TOKENS at the prompt price plus
MAX_COMPLETION_TOKENS at the completion price. A reported usage above those bounds is invalid: the larger amount is
kept as uncertain and the run stops.

Ledger is a locked, durable CSV journal that enforces, before every reservation, the batch cap, the cumulative
checkpoint (unless a valid confirmation file raises it) and the absolute technical stop. In the active configuration
(agent-budget-4) financial and attempt ceilings are None; accounting, reservations, uncertain amounts, reviewed
prices and request bounds remain enforced, and older finite configurations still load with their own limits.

The offline demo sends no request; its in-memory ledger (agent.ledger.MemoryLedger) still reads the reviewed prices
and the budget configuration defined here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo


MODEL = "nvidia/nemotron-3-super-120b-a12b"  # default model (Super), as in the live catalog
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"  # frozen config value, checked live at launch (catalog id and prices)
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"  # frozen config value, checked live at launch (catalog id and prices)
EXPECTED_IDS = (
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    MODEL,
    ULTRA,
    LIGHTNING,
)
MAX_COMPLETION_TOKENS = 16384  # all three models (agent-budget-3); 8000 in agent-budget-2, 4000 in -1
RESERVED_PROMPT_TOKENS = 60000  # worst-case prompt tokens of one request, enforced before send and at settlement
PROMPT_OVERHEAD_TOKENS = 2048  # chat-template allowance added to the UTF-8 byte count (prompt_token_bound)
MAX_MODEL_LEN = 262144  # Super's reviewed context length (catalog)
MESSAGE_KEYS = frozenset({"role", "content"})  # closed message schema
MESSAGE_ROLES = ("system", "user")


@dataclass(frozen=True)
class ModelPricing:
    """Frozen prices of one reviewed model (USD per TOKEN). max_model_len: the reviewed context length, or None
    when it was never reviewed (then a catalog value, if present, must be an int of at least
    RESERVED_PROMPT_TOKENS + MAX_COMPLETION_TOKENS)."""
    model: str
    prompt_price: Decimal
    completion_price: Decimal
    max_model_len: int | None
    label: str

    @property
    def pricing(self) -> tuple[Decimal, Decimal]:
        return self.prompt_price, self.completion_price

    @property
    def reservation_usd(self) -> Decimal:
        return RESERVED_PROMPT_TOKENS * self.prompt_price + MAX_COMPLETION_TOKENS * self.completion_price


REVIEWED_MODELS = {
    MODEL: ModelPricing(MODEL, Decimal("0.0000003"), Decimal("0.0000009"), MAX_MODEL_LEN, "Nemotron 3 Super"),
    ULTRA: ModelPricing(ULTRA, Decimal("0.000001"), Decimal("0.000003"), None, "Nemotron 3 Ultra"),
    LIGHTNING: ModelPricing(LIGHTNING, Decimal("0.00000006"), Decimal("0.00000024"), None, "Nemotron 3.5 Lightning"),
}
PROMPT_PRICE = REVIEWED_MODELS[MODEL].prompt_price  # Super (default model)
COMPLETION_PRICE = REVIEWED_MODELS[MODEL].completion_price
RESERVATION_USD = REVIEWED_MODELS[MODEL].reservation_usd  # Super: 60000 x 0.0000003 + 16384 x 0.0000009 = 0.0327456
LAB = Path(__file__).resolve().parent
BUDGET_CONFIG = LAB / "siteagent" / "config" / "agent_budget_v4.json"
NEVER_REACH_USD = Decimal("200")  # historical agent-budget-3 policy only
PARIS = ZoneInfo("Europe/Paris")
HEADER = ["date", "variant", "activity", "human_minutes", "expected_usd", "actual_usd", "notes"]
_EXTRA_PRICES = {"request", "image", "price_per_video_second", "price_per_minute"}
# Reviewed optional cache metadata. Null means no usable cache tariff; continue
# charging every prompt token at the full reviewed prompt rate, never at zero.
# A positive cache tariff is not accepted without a separate pricing review.
_NULLABLE_CACHE_PRICE = "input_cache_read"


def reviewed_model(model: str) -> ModelPricing:
    """The frozen pricing of a reviewed model; GuardrailError for any other id."""
    m = REVIEWED_MODELS.get(model) if isinstance(model, str) else None
    if m is None:
        raise GuardrailError("Model is not one of the reviewed models.")
    return m


def prompt_token_bound(messages: Any) -> int:
    """Bound of the prompt tokens of a chat request: UTF-8 bytes of every message content + 8 per message +
    PROMPT_OVERHEAD_TOKENS.

    A conservative byte-based HYPOTHESIS, not a proof: it assumes that every token of the endpoint's tokenizer covers at
    least one byte of content and that the chat template and role markers fit in 8 per message plus
    PROMPT_OVERHEAD_TOKENS. It was not verified against each endpoint's tokenizer, chat template or provider limit.
    Observed on recorded runs of three models: at most 13349 prompt tokens reported, with at least 13385 tokens of
    margin below this bound; an observation, not a guarantee. The second barrier stays
    at settlement: reported prompt_tokens above RESERVED_PROMPT_TOKENS are invalid usage (the larger amount is kept as
    uncertain and the run stops).

    The message schema is CLOSED. messages must be a non-empty list of dicts with exactly the keys role
    and content, role "system" or "user", content a str; anything else (an extra key such as name, which this bound
    would not count, the roles assistant or tool, non-text content) raises GuardrailError, so nothing is reserved
    (agent.loop) or sent (agent.real_client)."""
    if not isinstance(messages, list) or not messages:
        raise GuardrailError("Messages missing.")
    n = 0
    for m in messages:
        if not isinstance(m, dict) or set(m) != MESSAGE_KEYS:
            raise GuardrailError("Each message must have exactly the keys role and content.")
        if type(m["role"]) is not str or m["role"] not in MESSAGE_ROLES:
            raise GuardrailError("Message role must be system or user.")
        if type(m["content"]) is not str:
            raise GuardrailError("Message content must be text.")
        n += len(m["content"].encode("utf-8", "surrogatepass")) + 8
    return n + PROMPT_OVERHEAD_TOKENS


class GuardrailError(RuntimeError):
    """Stop without exposing request bodies, credentials, or raw exceptions."""


class BudgetError(GuardrailError):
    """A batch cap, the cumulative checkpoint or the absolute stop cannot cover the next request."""


def _amount(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise GuardrailError("Invalid non-negative amount.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise GuardrailError("Invalid non-negative amount.") from None
    if not result.is_finite() or result < 0:
        raise GuardrailError("Invalid non-negative amount.")
    return result


@dataclass(frozen=True)
class BudgetConfig:
    """Validated policy. None means no ceiling, never an infinite accounting amount."""
    version: str
    sha256: str
    batch_max_cost_usd: Decimal | None
    batch_max_calls: int | None
    batch_max_episodes: int
    batch_technical_retries: int | None
    max_stage_wall_s: float
    episode_max_cost_usd: Decimal | None
    episode_max_model_calls: int | None
    checkpoint_usd: Decimal | None
    absolute_stop_usd: Decimal | None
    confirmation_file: Path | None
    models: tuple = (MODEL,)  # agent-budget-2: the reviewed models this configuration prices
    max_completion_tokens: int = MAX_COMPLETION_TOKENS
    reserved_prompt_tokens: int = RESERVED_PROMPT_TOKENS


def _pos_int(value: Any, upper: int) -> int:
    if type(value) is not int or not 1 <= value <= upper:
        raise GuardrailError("Budget configuration: invalid count.")
    return value


def _check_models_section(data: dict) -> tuple:
    """agent-budget-2: request bounds and one pricing entry per reviewed model, each equal to REVIEWED_MODELS."""
    rb, models = data["request_bounds"], data["models"]
    if rb["max_completion_tokens"] != MAX_COMPLETION_TOKENS or rb["reserved_prompt_tokens"] != RESERVED_PROMPT_TOKENS \
            or rb["prompt_overhead_tokens"] != PROMPT_OVERHEAD_TOKENS:
        raise GuardrailError("Budget configuration: request bounds differ from the reviewed bounds.")
    if not isinstance(models, dict) or set(models) != set(REVIEWED_MODELS):
        raise GuardrailError("Budget configuration: the priced models differ from the reviewed models.")
    for mid, m in models.items():
        r = REVIEWED_MODELS[mid]
        if _amount(m["prompt_usd_per_token"]) != r.prompt_price or _amount(m["completion_usd_per_token"]) != r.completion_price \
                or _amount(m["reservation_per_call_usd"]) != r.reservation_usd or m.get("max_model_len") != r.max_model_len:
            raise GuardrailError("Budget configuration: pricing differs from the reviewed prices.")
    return tuple(sorted(models))


def load_budget_config(path: str | Path | None = None) -> BudgetConfig:
    """Read and check a versioned policy. agent-budget-4 accepts explicit null ceilings;
    agent-budget-3 retains finite historical caps. Missing or infinite amounts stop.
    agent-budget-1 (4000 output tokens, full-context reservation) and agent-budget-2 (8000 output tokens) are
    history: they no longer load (their request bound differs from MAX_COMPLETION_TOKENS)."""
    p = Path(path) if path is not None else BUDGET_CONFIG
    try:
        raw = p.read_bytes()
        data = json.loads(raw)
        b, e, c = data["batch"], data["episode"], data["cumulative"]
        models = _check_models_section(data)
        version = data["version"]
        if version not in ("agent-budget-3", "agent-budget-4"):
            raise GuardrailError("Budget configuration: unsupported version.")
        optional = version == "agent-budget-4"

        def cap(value):
            return None if optional and value is None else _amount(value)

        def count(value, upper):
            return None if optional and value is None else _pos_int(value, upper)

        retry = b["technical_retries"]
        if not (optional and retry is None) and (type(retry) is not int or not 0 <= retry <= 10):
            raise GuardrailError("Budget configuration: invalid retry count.")
        confirmation = c["confirmation_file"]
        if confirmation is None and optional:
            confirmation = None
        elif type(confirmation) is str and confirmation:
            confirmation = Path(confirmation)
        else:
            raise GuardrailError("Budget configuration: invalid confirmation path.")
        cfg = BudgetConfig(
            version=version, sha256=hashlib.sha256(raw).hexdigest(),
            batch_max_cost_usd=cap(b["max_cost_usd"]), batch_max_calls=count(b["max_calls"], 10_000),
            batch_max_episodes=_pos_int(b["max_episodes"], 10_000),
            batch_technical_retries=retry,
            max_stage_wall_s=float(_amount(b["max_stage_wall_s"])),
            episode_max_cost_usd=cap(e["max_cost_usd"]), episode_max_model_calls=count(e["max_model_calls"], 100),
            checkpoint_usd=cap(c["checkpoint_usd"]), absolute_stop_usd=cap(c["absolute_stop_usd"]),
            confirmation_file=confirmation, models=models)
    except GuardrailError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise GuardrailError("Budget configuration missing or unreadable.") from None
    if not math.isfinite(cfg.max_stage_wall_s) or cfg.max_stage_wall_s <= 0:
        raise GuardrailError("Budget configuration: invalid stage wall time.")
    worst_call = max(REVIEWED_MODELS[m].reservation_usd for m in cfg.models)
    finite_caps = [x for x in (cfg.episode_max_cost_usd, cfg.batch_max_cost_usd,
                              cfg.checkpoint_usd, cfg.absolute_stop_usd) if x is not None]
    if any(a > b for a, b in zip(finite_caps, finite_caps[1:])):
        raise GuardrailError("Budget configuration: finite cost caps must be ordered episode <= batch <= "
                             "checkpoint <= absolute stop.")
    if cfg.version == "agent-budget-3" and cfg.absolute_stop_usd >= NEVER_REACH_USD:
        raise GuardrailError("Budget configuration: agent-budget-3 requires absolute stop < 200 USD.")
    if cfg.episode_max_model_calls is not None and cfg.episode_max_cost_usd is not None \
            and worst_call * cfg.episode_max_model_calls > cfg.episode_max_cost_usd:
        raise GuardrailError("Budget configuration: the episode cost cap must cover episode calls x reservation.")
    # Review SA_R2 section 6: one money cap for the three models must never stop a model before the call cap
    if cfg.batch_max_calls is not None and cfg.batch_max_cost_usd is not None \
            and worst_call * cfg.batch_max_calls > cfg.batch_max_cost_usd:
        raise GuardrailError("Budget configuration: the batch cost cap must cover batch calls x reservation for every "
                             "reviewed model.")
    if cfg.checkpoint_usd is not None and cfg.confirmation_file is None:
        raise GuardrailError("Budget configuration: a cumulative checkpoint requires a confirmation path.")
    if cfg.confirmation_file is not None and (cfg.confirmation_file.is_absolute() or ".." in cfg.confirmation_file.parts):
        raise GuardrailError("Budget configuration: the confirmation file must be a path inside the lab.")
    return cfg


def cumulative_allowance(budget: BudgetConfig, confirmation_path: str | Path | None = None) -> tuple[Decimal | None, str]:
    """(highest cumulative spend allowed, why), with None meaning no ceiling.
    Without a checkpoint, no confirmation path is inspected. For a legacy checkpoint:
    checkpoint_usd unless the owner's confirmation file is valid (the format the code checks):
    {"confirmed_by": "Moad", "allow_cumulative_up_to_usd": "<amount>"} with checkpoint < amount <=
    absolute_stop_usd. Anything else (missing, malformed, other author, out of range) keeps the checkpoint."""
    if budget.checkpoint_usd is None:
        return budget.absolute_stop_usd, ("no cumulative ceiling; no confirmation required"
                                         if budget.absolute_stop_usd is None else "absolute stop only; no confirmation required")
    p = Path(confirmation_path) if confirmation_path is not None else LAB / budget.confirmation_file
    try:
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 4096:
            return budget.checkpoint_usd, "checkpoint (no confirmation file from Moad)"
        data = json.loads(p.read_text(encoding="utf-8"))
        amount = _amount(data["allow_cumulative_up_to_usd"])
        if data.get("confirmed_by") != "Moad" or amount <= budget.checkpoint_usd \
                or (budget.absolute_stop_usd is not None and amount > budget.absolute_stop_usd):
            return budget.checkpoint_usd, "checkpoint (confirmation file not valid)"
        return amount, "confirmed by Moad"
    except (OSError, ValueError, KeyError, TypeError, GuardrailError):
        return budget.checkpoint_usd, "checkpoint (confirmation file unreadable)"


def confirmation_info(budget: BudgetConfig, confirmation_path: str | Path | None = None) -> dict:
    """Launch metadata: no file access without a checkpoint. For legacy finite policies, report the
    confirmation's date, amount and acceptance. This is a POLICY, not authentication: any local process
    could write the file, but only the project owner is authorized to do so for those historical policies."""
    if budget.checkpoint_usd is None:
        limit, why = cumulative_allowance(budget)
        return {"path": None, "present": None, "required": False, "checked": False, "accepted": False,
                "allowed_cumulative_usd": None if limit is None else str(limit), "reason": why}
    p = Path(confirmation_path) if confirmation_path is not None else LAB / budget.confirmation_file
    limit, why = cumulative_allowance(budget, p)
    info = {"path": str(budget.confirmation_file), "present": p.is_file() and not p.is_symlink(),
            "accepted": why == "confirmed by Moad", "allowed_cumulative_usd": str(limit)}
    if info["present"]:
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.stat().st_size <= 4096 else {}
            info.update(confirmed_by=str(data.get("confirmed_by"))[:20], date=str(data.get("date"))[:20],
                        amount=str(data.get("allow_cumulative_up_to_usd"))[:20],
                        file_modified=datetime.fromtimestamp(p.stat().st_mtime, PARIS).isoformat(timespec="seconds"))
        except (OSError, ValueError, AttributeError):
            info["unreadable"] = True
    return info


def validate_pricing(model_dict: dict, model: str = MODEL) -> tuple[Decimal, Decimal]:
    """Accept only the reviewed model `model` and its frozen USD-per-token prices (REVIEWED_MODELS)."""
    r = reviewed_model(model)
    if not isinstance(model_dict, dict) or model_dict.get("id") != r.model:
        raise GuardrailError("The reviewed model was not found.")
    for field in ("max_model_len", "context_length"):
        if field in model_dict:
            v = model_dict[field]
            if r.max_model_len is not None and (type(v) is not int or v != r.max_model_len):
                raise GuardrailError("Model context length changed; review required.")
            if r.max_model_len is None and (type(v) is not int or v < RESERVED_PROMPT_TOKENS + MAX_COMPLETION_TOKENS):
                raise GuardrailError("Model context length below the reviewed request bound; review required.")
    pricing = model_dict.get("pricing")
    if not isinstance(pricing, dict) or not {"prompt", "completion"} <= pricing.keys():
        raise GuardrailError("Missing model pricing.")
    if pricing.keys() - {"prompt", "completion", _NULLABLE_CACHE_PRICE} - _EXTRA_PRICES:
        raise GuardrailError("Unknown pricing field; review required.")
    ordinary = {name: value for name, value in pricing.items() if name != _NULLABLE_CACHE_PRICE}
    if any(not isinstance(value, str) for value in ordinary.values()):
        raise GuardrailError("Pricing values must be decimal strings.")
    if _NULLABLE_CACHE_PRICE in pricing and pricing[_NULLABLE_CACHE_PRICE] is not None:
        cached = pricing[_NULLABLE_CACHE_PRICE]
        if not isinstance(cached, str) or _amount(cached) != 0:
            raise GuardrailError("Cache tariff requires review; no discount assumed.")
    parsed = {name: _amount(value) for name, value in ordinary.items()}
    if (parsed["prompt"], parsed["completion"]) != r.pricing:
        raise GuardrailError("Token prices changed; review required.")
    if any(parsed.get(name, Decimal(0)) != 0 for name in _EXTRA_PRICES):
        raise GuardrailError("Additional charges require review.")
    return parsed["prompt"], parsed["completion"]


def final_text(content: str | None) -> str:
    """Keep only the answer after a completed reasoning block, if present."""
    if not isinstance(content, str):
        raise GuardrailError("Missing model answer.")
    answer = content.split("</think>", 1)[-1].strip()
    if not answer or re.search(r"</?think\b", answer, flags=re.IGNORECASE):
        raise GuardrailError("Empty answer or unfinished reasoning block.")
    return answer


def _now() -> datetime:
    return datetime.now(PARIS)


def _row_day(value: str) -> date:
    try:
        if "T" not in value and " " not in value:
            return date.fromisoformat(value)
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            raise ValueError
        return stamp.astimezone(PARIS).date()
    except (TypeError, ValueError):
        raise GuardrailError("Invalid ledger date.") from None


class Ledger:
    """Lock the existing CSV for a whole run and persist costs before requests.

    An unsettled row keeps its full reservation. This ceiling only covers clients
    using this same journal; it does not cover Playground or other API clients.
    Row indices are zero-based data-row indices, excluding the CSV header.
"""

    def __init__(self, prep_log_path: str | Path, budget: BudgetConfig | None = None, batch_id: str | None = None,
                 confirmation_path: str | Path | None = None):
        """batch_id: rows of every run of one batch (its stages, possibly in several processes) share the
        batch caps; without it, the batch caps apply to this run's own rows."""
        self.path = Path(prep_log_path)
        self.budget = budget if budget is not None else load_budget_config()
        if batch_id is not None and not re.fullmatch(r"[A-Za-z0-9_\-]{3,80}", batch_id):
            raise GuardrailError("Invalid batch id.")
        self.batch_id = batch_id
        self._confirmation = confirmation_path
        self._file = None
        self._rows: list[dict[str, str]] = []
        self._own_rows: set[int] = set()
        self._models: dict[int, str] = {}
        self._run_id = uuid4().hex
        self._used = False

    def __enter__(self) -> Ledger:
        if self._used:
            raise GuardrailError("A ledger instance can only be entered once.")
        self._used = True
        try:
            # A separate lock file or replacing this inode would weaken flock.
            fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            self._file = os.fdopen(fd, "r+", encoding="utf-8", newline="")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            parsed = list(csv.reader(self._file, strict=True))
            if not parsed or parsed[0] != HEADER:
                raise GuardrailError("Ledger header is missing or different.")
            if any(len(row) != len(HEADER) for row in parsed[1:]):
                raise GuardrailError("Malformed ledger row.")
            self._rows = [dict(zip(HEADER, row)) for row in parsed[1:]]
            for row in self._rows:
                _row_day(row["date"])
                for field in ("human_minutes", "expected_usd", "actual_usd"):
                    if row[field] != "":
                        _amount(row[field])
                if row["variant"] == "token_factory":
                    if row["expected_usd"] == "" or _amount(row["expected_usd"]) <= 0:
                        raise GuardrailError("Missing Token Factory reservation.")
            self._check_budgets()
            return self
        except GuardrailError:
            self._close()
            raise
        except (OSError, UnicodeError, csv.Error):
            self._close()
            raise GuardrailError("Ledger unavailable, locked, or unreadable.") from None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._close()
        return False

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _require_open(self) -> None:
        if self._file is None:
            raise GuardrailError("Use the ledger inside its context manager.")

    def _cost(self, row: dict[str, str]) -> Decimal:
        return _amount(row["actual_usd"] if row["actual_usd"] != "" else row["expected_usd"])

    def _tag(self) -> str:
        return f"run={self._run_id};" + (f"batch={self.batch_id};" if self.batch_id else "")

    def _scope_rows(self) -> list[dict[str, str]]:
        """Rows counted against the batch caps: the batch's rows (any run), else this run's rows."""
        if self.batch_id:
            tag = f"batch={self.batch_id}"
            return [row for row in self._rows if row["variant"] == "token_factory" and tag in row["notes"].split(";")]
        return [self._rows[i] for i in sorted(self._own_rows)]

    def _cumulative(self) -> Decimal:
        return sum((self._cost(row) for row in self._rows if row["variant"] == "token_factory"), Decimal(0))

    def totals(self) -> dict[str, Any]:
        """Committed amounts (settled + kept reservations), for the run summary. No secret inside."""
        scope = self._scope_rows()
        limit, why = cumulative_allowance(self.budget, self._confirmation)
        return {"budget_version": self.budget.version, "batch_id": self.batch_id,
                "cumulative_committed_usd": str(self._cumulative()),
                "cumulative_allowed_usd": None if limit is None else str(limit),
                "cumulative_allowed_why": why,
                "absolute_stop_usd": None if self.budget.absolute_stop_usd is None else str(self.budget.absolute_stop_usd),
                "confirmation_file": confirmation_info(self.budget, self._confirmation),
                "batch_committed_usd": str(sum((self._cost(r) for r in scope), Decimal(0))),
                "batch_calls": len(scope),
                "batch_max_cost_usd": None if self.budget.batch_max_cost_usd is None else str(self.budget.batch_max_cost_usd),
                "batch_max_calls": self.budget.batch_max_calls}

    def _check_budgets(self, additional: Decimal = Decimal(0), calls: int = 0) -> None:
        cumulative = self._cumulative()
        if self.budget.absolute_stop_usd is not None and cumulative + additional > self.budget.absolute_stop_usd:
            raise BudgetError("The absolute technical stop (cumulative Token Factory spend) would be exceeded.")
        limit, _why = cumulative_allowance(self.budget, self._confirmation)
        if limit is not None and cumulative + additional > limit:
            if limit == self.budget.checkpoint_usd:
                raise BudgetError("Cumulative checkpoint reached: Moad's confirmation file is required to go on.")
            raise BudgetError("The cumulative amount confirmed by Moad would be exceeded.")
        scope = self._scope_rows()
        if self.budget.batch_max_cost_usd is not None \
                and sum((self._cost(r) for r in scope), Decimal(0)) + additional > self.budget.batch_max_cost_usd:
            raise BudgetError("The batch Token Factory cost cap is exhausted.")
        if self.budget.batch_max_calls is not None and len(scope) + calls > self.budget.batch_max_calls:
            raise BudgetError("The batch call cap is reached.")

    def _save(self) -> None:
        self._require_open()
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self._rows)
        try:
            # Keep the locked inode. Write before truncating so a partial write
            # does not first discard all outstanding reservations.
            self._file.seek(0)
            self._file.write(buffer.getvalue())
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.truncate()
            self._file.flush()
            os.fsync(self._file.fileno())
        except (OSError, UnicodeError):
            self._close()
            raise GuardrailError("Ledger write failed; no further request is allowed.") from None

    def reserve(self, activity: str, expected_usd: Decimal, model: str = MODEL) -> int:
        """model: the reviewed model the request goes to; the reservation may not be below its bound."""
        self._require_open()
        expected = _amount(expected_usd)
        if expected < reviewed_model(model).reservation_usd:
            raise GuardrailError("Reservation is below the reviewed request bound.")
        if not isinstance(activity, str) or not activity.strip():
            raise GuardrailError("A reservation requires an activity name.")
        self._check_budgets(expected, calls=1)
        index = len(self._rows)
        self._rows.append({
            "date": _now().isoformat(timespec="seconds"),
            "variant": "token_factory",
            "activity": activity,
            "human_minutes": "",
            "expected_usd": str(expected),
            "actual_usd": "",
            "notes": (f"{self._tag()}model={model};status=uncertain;"
                      "expected_usd=conservative_reservation_not_prompt_forecast"),
        })
        self._models[index] = model
        self._own_rows.add(index)
        self._save()
        return index

    def settle(self, row: int, usage_dict: dict, pricing: tuple[Decimal, Decimal], latency_s: float) -> Decimal:
        """pricing must be the frozen prices of the model the row was reserved for."""
        self._require_open()
        if type(row) is not int or row not in self._own_rows:
            raise GuardrailError("This reservation belongs to another run.")
        entry = self._rows[row]
        if entry["actual_usd"] != "":
            raise GuardrailError("Reservation already settled.")
        model = self._models.get(row, MODEL)
        r = reviewed_model(model)
        if pricing != r.pricing:
            raise GuardrailError("Unreviewed pricing.")
        if not isinstance(usage_dict, dict):
            raise GuardrailError("Usage unavailable; full cost reservation retained.")
        fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(type(usage_dict.get(name)) is not int or usage_dict[name] < 0 for name in fields):
            raise GuardrailError("Invalid usage; full cost reservation retained.")
        usage = {name: usage_dict[name] for name in fields}
        prompt, completion, total = (usage[name] for name in fields)
        cost = prompt * pricing[0] + completion * pricing[1]
        valid = (total == prompt + completion and prompt <= RESERVED_PROMPT_TOKENS
                 and completion <= MAX_COMPLETION_TOKENS
                 and (r.max_model_len is None or total <= r.max_model_len))
        if not valid:
            # Plausible extra charges must not disappear even when counts are
            # inconsistent. Keep the larger bound as unknown, then stop.
            entry["expected_usd"] = str(max(_amount(entry["expected_usd"]), cost))
            entry["notes"] = (f"{self._tag()}model={model};status=uncertain;reason=invalid_usage;"
                              "expected_usd=conservative_reservation_not_prompt_forecast")
            self._save()
            raise GuardrailError("Usage exceeds the reviewed bounds or is inconsistent.")
        latency = _amount(latency_s)
        entry["actual_usd"] = str(cost)
        entry["notes"] = (f"{self._tag()}model={model};status=settled;latency_s={latency};"
                          "expected_usd=conservative_reservation_not_prompt_forecast;"
                          f"prompt_tokens={prompt};completion_tokens={completion};total_tokens={total}")
        self._save()
        print(json.dumps(usage, sort_keys=True), file=sys.stderr)
        self._check_budgets()
        return cost
