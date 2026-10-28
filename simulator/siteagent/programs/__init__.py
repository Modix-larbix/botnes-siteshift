"""Task programs for "find the sponge, pick it, scrub the target bowl", and the conditions they stand for.

Every program receives the same skill facade (programs.facade: same skills, same calibration, same episode budget)
and is scored by the same bench-side evaluator (siteagent.taskcheck). Programs are not given the simulator, the seed,
the evaluator or its verdict as arguments, but an in-process program is not isolated from them; the agent's programs
run in the sandbox (agent.sandbox).

- Condition A, programs.fixed_reference: written once for the reference kitchen and its correct dossier, with
  positions measured once at installation (exact simulator truth, which favors A); no perception, no search.
- Condition B, a dossier-guided program without exploration: defined as a comparator, not built here.
- The scripted adaptive chain, programs.scripted_chain: a programmed search, no language model; the control that tells
  a gain of the agent apart from a gain of plain programmed search.
- Condition C, the agent loop (siteagent.agent): a model client writes and corrects the program from observable
  feedback. In the offline demo the client is the demo client, which serves fixed programs and a recorded choice.
"""
