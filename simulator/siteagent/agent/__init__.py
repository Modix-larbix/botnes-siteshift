"""Agent side of the Site Agent: bounded programs, their interpreter and sandbox, the agent loop, budgets.

Modules (import them directly; this file imports nothing so the sandboxed child process stays light):
  limits        program, episode and agent limits (agent_limits_s5_v0_8.json)
  api           skill API seen by programs (names, argument checks, description for the prompt)
  lang          BotScript: program text to a checked plain form (never compiled or run by Python)
  interp        interpreter of the checked form (plain values only, limits before every step)
  sandbox       separate locked-down process for a program run (with sandbox_child)
  host          GuardedHost (episode limits, re-checks) and its backends (StubKitchen, FacadeBackend)
  prompt        prompt builder with leak guard, response parser, observable run report
  llm           response type, mock client, request body and fingerprint
  real_client   guarded client for a real model endpoint (not used by the offline demo)
  ledger        cost reservation and settlement
  memory        program library per site, assumptions, invalidation
  loop          the agent loop
  bench_agent   the loop wired to the simulator and the evaluator
"""
