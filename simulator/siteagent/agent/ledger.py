"""Cost and usage ledger of the agent: reserve before a model call, settle after.

Interface used by the agent loop:
  reserve(activity) -> token        raises BudgetStop when the next call could exceed a cap
  consume_for_send(token)           binds one send to that reservation (refused when unknown, closed or used)
  settle(token, response, version)  computed cost (Decimal) from the provider's usage and the reviewed prices
  keep_uncertain(token, reason)     a failed request keeps its full reservation as uncertain cost
  entries                           one row per call: model id, prompt version, tokens, latency, cost

Caps come from the budget configuration read by tf_guardrails.load_budget_config (the source shared with
tf_guardrails.Ledger). In the active configuration (agent-budget-4) None means no ceiling; reservations, counters and
uncertain costs remain, and older finite configurations still load and are enforced. A reservation is the reviewed
model's bounded worst case (RESERVED_PROMPT_TOKENS prompt tokens plus the completion limit at its prices); usage
above those bounds is invalid and kept as uncertain cost.

- MemoryLedger: in-memory accounting with the same prices and caps, for mock runs, dry runs, tests and the offline
  demo. It spends nothing: computed costs are what a call would cost at the reviewed prices.
- GuardedLedger: wraps a tf_guardrails.Ledger (a locked, durable CSV journal) that enforces the file-backed caps.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from tf_guardrails import (COMPLETION_PRICE, MAX_COMPLETION_TOKENS, PROMPT_PRICE, RESERVATION_USD,  # noqa: F401
                           RESERVED_PROMPT_TOKENS, REVIEWED_MODELS, BudgetConfig, ModelPricing, load_budget_config)

from .llm import ClientNotReady

MAX_MODEL_LEN = 262144  # Super's reviewed context length (tf_guardrails.MAX_MODEL_LEN)


def model_pricing(model: str | None = None) -> ModelPricing:
    """Frozen pricing of a reviewed model (default: Super). ValueError for any other id."""
    m = REVIEWED_MODELS.get(model or "nvidia/nemotron-3-super-120b-a12b")
    if m is None:
        raise ValueError("not a reviewed model")
    return m


class BudgetStop(RuntimeError):
    """The next call could exceed a cap: no request is made."""


def computed_cost(usage: dict, pricing: tuple | None = None) -> Decimal:
    p, c = pricing or (PROMPT_PRICE, COMPLETION_PRICE)
    return int(usage["prompt_tokens"]) * p + int(usage["completion_tokens"]) * c


def uncertain_amount(usage, reservation: Decimal, pricing: tuple | None = None) -> Decimal:
    """Amount kept as uncertain when the reported usage is outside the reviewed bounds: the larger of
    the reservation and the cost at the reported counts, when those counts are non-negative
    integers (the rule of tf_guardrails.Ledger.settle); else the reservation."""
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys):
        return max(Decimal(reservation), computed_cost(usage, pricing))
    return Decimal(reservation)


def _entry_cost(e: dict, reservation: Decimal) -> Decimal:
    if e["cost_usd"] is not None:
        return Decimal(e["cost_usd"])
    return Decimal(e.get("uncertain_usd") or reservation)


def consume_reservation(entries: list, token) -> dict:
    """Bind one send to one reservation, BEFORE the transport. The reservation must
    exist, be open and never used for a send; it stays consumed even if the transport then fails."""
    if type(token) is not int or not 0 <= token < len(entries):
        raise ClientNotReady("unknown reservation id; no request made")
    e = entries[token]
    if e.get("status") != "reserved":
        raise ClientNotReady("reservation already closed; no request made")
    if e.get("sent"):
        raise ClientNotReady("reservation already used for a send; one send per reservation; no request made")
    e["sent"] = True
    return e


def _valid_usage(u, max_model_len: int | None = MAX_MODEL_LEN) -> bool:
    """Same bounds as tf_guardrails.Ledger.settle: prompt <= RESERVED_PROMPT_TOKENS (the reservation's bound),
    completion <= MAX_COMPLETION_TOKENS, total = prompt + completion, total <= the reviewed context if known."""
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return (isinstance(u, dict) and all(type(u.get(k)) is int and u[k] >= 0 for k in keys)
            and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
            and u["prompt_tokens"] <= RESERVED_PROMPT_TOKENS and u["completion_tokens"] <= MAX_COMPLETION_TOKENS
            and (max_model_len is None or u["total_tokens"] <= max_model_len))


@dataclass
class MemoryLedger:
    """In-memory ledger using the active budget. run_limit / max_calls default to its
    optional batch caps. confirmed_up_to_usd simulates the legacy confirmation (tests only)."""
    prior_cumulative_usd: Decimal = Decimal(0)
    budget: BudgetConfig = field(default_factory=load_budget_config)
    run_limit: Decimal | None = None
    max_calls: int | None = None
    confirmed_up_to_usd: Decimal | None = None
    label: str = "mock: no money spent; costs computed at the reviewed Token Factory prices"
    entries: list = field(default_factory=list)
    model: str = "nvidia/nemotron-3-super-120b-a12b"  # the reviewed model this ledger prices

    def __post_init__(self):
        self.pricing_of = model_pricing(self.model)
        self.reservation_usd = self.pricing_of.reservation_usd
        self.run_limit = self.budget.batch_max_cost_usd if self.run_limit is None else Decimal(self.run_limit)
        self.max_calls = self.budget.batch_max_calls if self.max_calls is None else self.max_calls
        if self.run_limit is not None and (not self.run_limit.is_finite() or self.run_limit < 0
                or (self.budget.absolute_stop_usd is not None and self.run_limit > self.budget.absolute_stop_usd)):
            raise ValueError("MemoryLedger cost cap must be finite, non-negative and under any absolute stop")
        if self.max_calls is not None and (type(self.max_calls) is not int or self.max_calls < 1):
            raise ValueError("MemoryLedger call cap must be a positive integer or None")

    def _run_cost(self) -> Decimal:
        return sum((_entry_cost(e, Decimal(e["reserved_usd"])) for e in self.entries), Decimal(0))

    def allowed_cumulative(self) -> Decimal | None:
        if self.budget.checkpoint_usd is None:
            return self.budget.absolute_stop_usd
        c = self.confirmed_up_to_usd
        if c is not None and Decimal(c).is_finite() and self.budget.checkpoint_usd < Decimal(c) \
                and (self.budget.absolute_stop_usd is None or Decimal(c) <= self.budget.absolute_stop_usd):
            return Decimal(c)
        return self.budget.checkpoint_usd

    def reserve(self, activity: str) -> int:
        run = self._run_cost()
        b = self.budget
        if self.max_calls is not None and len(self.entries) + 1 > self.max_calls:
            raise BudgetStop(f"batch call cap ({self.max_calls}) reached")
        res = self.reservation_usd
        if self.run_limit is not None and run + res > self.run_limit:
            raise BudgetStop(f"batch cap ({self.run_limit} USD) would be exceeded")
        cumulative = self.prior_cumulative_usd + run + res
        if b.absolute_stop_usd is not None and cumulative > b.absolute_stop_usd:
            raise BudgetStop(f"absolute technical stop ({b.absolute_stop_usd} USD cumulative) would be exceeded")
        allowed = self.allowed_cumulative()
        if allowed is not None and cumulative > allowed:
            raise BudgetStop(f"cumulative checkpoint ({allowed} USD) would be exceeded: "
                             "Moad's confirmation is required to go on")
        self.entries.append({"activity": activity, "reserved_usd": str(res), "cost_usd": None,
                             "status": "reserved", "model": self.model})
        return len(self.entries) - 1

    def consume_for_send(self, token) -> None:
        consume_reservation(self.entries, token)

    def settle(self, token: int, response, prompt_version: str) -> Decimal:
        e = self.entries[token]
        if e["status"] != "reserved":
            raise BudgetStop("reservation already closed")
        pr = self.pricing_of
        if not _valid_usage(response.usage, pr.max_model_len):
            amt = uncertain_amount(response.usage, Decimal(e["reserved_usd"]), pr.pricing)
            e.update(status="uncertain", uncertain_usd=str(amt),
                     note="invalid usage; max(reservation, cost at reported usage) kept")
            raise BudgetStop("invalid usage reported; stopped with the larger of the reservation and the reported cost")
        cost = computed_cost(response.usage, pr.pricing)
        e.update(status="settled", cost_usd=str(cost), model_id=response.model_id, prompt_version=prompt_version,
                 usage=dict(response.usage), usage_source=response.usage_source, latency_s=response.latency_s)
        return cost

    def keep_uncertain(self, token: int, reason: str) -> None:
        self.entries[token].update(status="uncertain", note=reason)

    def has_open_reservation(self) -> bool:
        """A reservation made and not yet settled or closed (real_client sends only then)."""
        return any(e["status"] == "reserved" for e in self.entries)

    def summary(self) -> dict:
        settled = [e for e in self.entries if e["status"] == "settled"]
        return {"label": self.label, "budget_version": self.budget.version, "model": self.model,
                "reservation_per_call_usd": str(self.reservation_usd),
                "batch_cap_usd": None if self.run_limit is None else str(self.run_limit),
                "batch_max_calls": self.max_calls, "calls": len(self.entries), "settled": len(settled),
                "uncertain": sum(e["status"] == "uncertain" for e in self.entries),
                "computed_cost_usd": str(sum((Decimal(e["cost_usd"]) for e in settled), Decimal(0))),
                "reserved_uncertain_usd": str(sum((_entry_cost(e, Decimal(e["reserved_usd"])) for e in self.entries
                                                   if e["status"] == "uncertain"), Decimal(0)))}


def cumulative_token_factory_usd(csv_path: str | Path) -> Decimal:
    """Sum of Token Factory rows of a prep_log.csv (actual if settled, else the kept reservation)."""
    total = Decimal(0)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant") == "token_factory":
                total += Decimal(row["actual_usd"] if row["actual_usd"] != "" else row["expected_usd"])
    return total


class GuardedLedger:
    """Adapter over an ENTERED tf_guardrails.Ledger: locked, durable CSV journal (prep_log.csv) that
    checks any explicitly configured caps before each reservation. Active agent-budget-4 has none; explicit
    historical agent-budget-3 still enforces its finite ceilings. Usage: with tf_guardrails.Ledger(path,
    batch_id=...) as L: g = GuardedLedger(L, pricing, reservation, model). model: the reviewed model of the run;
    pricing must be its frozen prices and the reservation at least its bounded worst case."""

    def __init__(self, tf_ledger, pricing: tuple, reservation_usd: Decimal, model: str | None = None):
        self._l, self._pricing = tf_ledger, pricing
        self.pricing_of = model_pricing(model)
        self.model = self.pricing_of.model
        self._res = Decimal(reservation_usd)
        if tuple(pricing) != self.pricing_of.pricing:
            raise ValueError("pricing is not the frozen pricing of the model")
        if self._res < self.pricing_of.reservation_usd:
            raise ValueError("the reservation is below the reviewed worst case of one call")
        self.reservation_usd = self._res
        self.entries: list[dict] = []

    def _own(self) -> Decimal:
        return sum((_entry_cost(e, self._res) for e in self.entries), Decimal(0))

    def reserve(self, activity: str) -> int:
        try:
            row = self._l.reserve(activity, self._res, model=self.model)
        except Exception as e:  # tf_guardrails.BudgetError / GuardrailError: its literal message only
            msg = str(e) if type(e).__name__ in ("BudgetError", "GuardrailError") else ""
            raise BudgetStop(f"tf_guardrails refused the reservation ({type(e).__name__}: {msg[:160]})") from None
        self.entries.append({"activity": activity, "row": row, "reserved_usd": str(self._res), "cost_usd": None,
                             "status": "reserved", "model": self.model})
        return len(self.entries) - 1

    def settle(self, token: int, response, prompt_version: str) -> Decimal:
        e = self.entries[token]
        try:
            cost = self._l.settle(e["row"], response.usage, self._pricing, response.latency_s)
        except Exception as err:
            # Same amount as the CSV row: tf_guardrails keeps max(reservation, cost at reported usage).
            e.update(status="uncertain", note=type(err).__name__,
                     uncertain_usd=str(uncertain_amount(response.usage, self._res, self._pricing)))
            raise BudgetStop("settlement refused; the larger of the reservation and the reported cost is kept") from None
        e.update(status="settled", cost_usd=str(cost), model_id=response.model_id, prompt_version=prompt_version,
                 usage=dict(response.usage), usage_source=response.usage_source, latency_s=response.latency_s)
        return cost

    def keep_uncertain(self, token: int, reason: str) -> None:
        self.entries[token].update(status="uncertain", note=reason)  # the CSV row keeps its reservation

    def has_open_reservation(self) -> bool:
        return any(e["status"] == "reserved" for e in self.entries)

    def consume_for_send(self, token) -> None:
        consume_reservation(self.entries, token)

    def summary(self) -> dict:
        settled = [e for e in self.entries if e["status"] == "settled"]
        return {"label": "prep_log.csv journal (tf_guardrails.Ledger)", "model": self.model,
                "reservation_per_call_usd": str(self._res), "calls": len(self.entries),
                "settled": len(settled), "uncertain": sum(e["status"] == "uncertain" for e in self.entries),
                "computed_cost_usd": str(sum((Decimal(e["cost_usd"]) for e in settled), Decimal(0))),
                "reserved_uncertain_usd": str(sum((_entry_cost(e, self._res) for e in self.entries
                                                   if e["status"] == "uncertain"), Decimal(0))),
                "journal": self._l.totals()}
