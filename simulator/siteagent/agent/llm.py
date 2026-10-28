"""Model client interface of the agent. No network here.

- LLMResponse: what the agent loop needs from one call (text, finish reason, usage, model id, latency, and where the
  usage numbers come from).
- MockLLMClient: scripted answers for tests; no network, no money; its token counts are a deterministic estimate
  labelled "mock_estimate". A mock run says nothing about a real model.
- build_request: the exact request body a real call would send (model id, messages, temperature 0,
  max_completion_tokens, and reasoning_effort only when it is not None and one of REASONING_EFFORTS).
- request_fingerprint: sha256 of that body; the offline demo serves a recorded answer only for the same fingerprint.
- TokenFactoryClient: a placeholder that refuses to send; the guarded client is agent.real_client.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import ClassVar

MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"  # tf_guardrails.MODEL (a test checks they match)
# Token Factory chat API reference (documented values of reasoning_effort); per-model support untested
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class TransientLLMError(RuntimeError):
    """A technical failure of the request (timeout, 429, 5xx): may be retried at most once per run."""


class LLMRefused(RuntimeError):
    """Truncated, refused or empty completion: not retried technically; counted as an attempt."""


class ClientNotReady(RuntimeError):
    """A precondition of a real call is missing (preflight, reservation, request shape, run
    authorization or its envelope): nothing is sent. Defined here so the loop can catch it without
    importing the real client (agent.real_client re-exports it)."""


@dataclass
class LLMResponse:
    text: str
    finish_reason: str
    usage: dict
    model_id: str
    latency_s: float
    usage_source: str  # "provider" (real call) | "mock_estimate" | "fake_transport" | "unknown_transport"
    reasoning: str | None = None  # a separate reasoning field of the provider, kept bench side only


def valid_reasoning_effort(value) -> bool:
    """A documented reasoning_effort value (a plain str in REASONING_EFFORTS). None is not a value: it means 'not sent'."""
    return type(value) is str and value in REASONING_EFFORTS


def build_request(messages: list[dict], max_completion_tokens: int, temperature: float = 0.0,
                  model: str = MODEL_ID, reasoning_effort: str | None = None) -> dict:
    """reasoning_effort: None (default) sends no such key; otherwise one documented value (ValueError else)."""
    req = {"model": model, "messages": messages, "temperature": temperature,
           "max_completion_tokens": int(max_completion_tokens)}
    if reasoning_effort is not None:
        if not valid_reasoning_effort(reasoning_effort):
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS} (or None: not sent)")
        req["reasoning_effort"] = reasoning_effort
    return req


def request_fingerprint(req: dict) -> str:
    payload = json.dumps(req, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class MockLLMClient:
    """answers: list of strings (returned in order) or a function(messages, n) -> str.
    fail_on: call numbers (1-based) that raise TransientLLMError instead of answering."""
    client_kind: ClassVar[str] = "mock"  # labels: client simule (bench_agent.client_label)
    answers: object
    model_id: str = "mock/no-model"
    fail_on: tuple = ()
    finish_reason: str = "stop"
    received: list = field(default_factory=list)
    n: int = 0

    def complete(self, request: dict) -> LLMResponse:
        self.n += 1
        self.received.append(json.loads(json.dumps(request)))  # a copy of exactly what was "sent"
        t0 = time.perf_counter()
        if self.n in self.fail_on:
            raise TransientLLMError("mock transient failure")
        if callable(self.answers):
            text = self.answers(request["messages"], self.n)
        else:
            k = sum(1 for i in range(1, self.n + 1) if i not in self.fail_on) - 1
            if k >= len(self.answers):
                raise LLMRefused("mock has no more answers")
            text = self.answers[k]
        p = sum(len(m["content"]) for m in request["messages"]) // 4
        c = min(len(text) // 4, request["max_completion_tokens"])
        return LLMResponse(text=text, finish_reason=self.finish_reason,
                           usage={"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
                           model_id=self.model_id, latency_s=round(time.perf_counter() - t0, 6),
                           usage_source="mock_estimate")


class TokenFactoryClient:  # pragma: no cover (placeholder on purpose)
    def __init__(self, *a, **kw):
        raise NotImplementedError("Real Token Factory calls are not wired in S5 preparation (track 2). "
                                  "See verif/S5_PREP_NOTES.md, real-call protocol; Moad approves first.")
