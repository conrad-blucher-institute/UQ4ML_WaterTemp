# Resume notes — `esb_dev_auto_agent` branch

Snapshot so future-you (or a fresh Claude session) can pick up without re-deriving context.

## What this branch is
A new branch `esb_dev_auto_agent` cut from `esb_dev` to build a parallel
agent-orchestration system. Plan was decided after comparing three drafts
(Haiku 4.5, Sonnet 4.6, Opus 4.6) — went with **Sonnet 4.6's brief + Opus
4.6's additions**: Claude Code CLI as the agent runner (subprocess, not SDK),
JSONL logs, stdlib + pyyaml only, model is a swappable string field.

## What exists on disk right now
- `orchestrator/BRIEF.md` — merged brief (was already present)
- `orchestrator/tasks.yaml` — two dummy tasks (`dummy_docs`, `dummy_optimize`) targeting `claude-haiku-4-5-20251001`
- `orchestrator/prompts/docs.md` — dummy docs prompt

## What is MISSING (writes silently failed mid-session, must be recreated)
- `orchestrator/orchestrator.py` — the async runner. Design was: asyncio + subprocess to `claude` CLI, one git worktree per task under `orchestrator/agent_logs/worktrees/<task>/`, JSONL stream log per task, REPORT.md at end, auto-push branches but do NOT auto-create PRs (gh is not authed on this laptop).
- `orchestrator/prompts/optimize.md` — dummy optimize prompt (mirror of docs.md)
- `orchestrator/prompts/refactor.md` — template prompt with empty scope
- `orchestrator/prompts/ml_train.md` — template prompt, flagged as risky for parallel exec
- `orchestrator/README.md` — usage doc
- `orchestrator/.gitignore` — ignore `agent_logs/` and `agent_output/`

## Key design decisions already locked in
- Agents invoked via `claude` CLI subprocess (not Anthropic SDK direct).
- Each agent runs in its own git worktree so parallel branches don't fight over the working dir.
- `permission-mode: acceptEdits` and `--output-format stream-json --verbose` on the claude invocation.
- Branch naming: `agent/<task-name>`.
- Each agent makes its own commits; orchestrator pushes but does not open PRs.

## Repo-integration notes (for when we wire to real files, NOT today)
- Plausible tracks: docs on `src/helper/utils*.py`, refactor reconciling `src/driver/crps_mme_runner.py` vs `crps_mme_runner copy.py`, cleanup of root-level `test*.py` / `verify_fix.py`, optimize pass on `crps` / `crps_loss`, ml_train on a single driver.
- **Do NOT scope agents to `src/helper/utils_mse_crps.py`** until the `dev-Proto_Incorp` stash on this laptop is resolved — it has unmerged decorator changes.
- `src/driver/` files import shared helpers — non-overlapping file scopes are critical or branches will conflict at merge.
- `ml_train` agents that produce `.keras` files: gitignore on the agent branch or push to LFS, don't commit raw binaries.

## Outstanding tasks when resuming
1. Recreate the six missing files listed above.
2. `git add orchestrator/ && git commit` with a message like `feat(agent): scaffold orchestrator skeleton`.
3. `git push -u origin esb_dev_auto_agent`.
4. Print the GitHub compare URL (`https://github.com/conrad-blucher-institute/UQ4ML_WaterTemp/compare/esb_dev...esb_dev_auto_agent`) so the user can open the PR in browser — `gh` is not authenticated on this laptop.
5. Walk the user through the diff.

## Environment gotchas
- `gh auth status` → not logged in. PR creation via CLI is not available; use compare URL.
- Shell is PowerShell. Use PowerShell syntax for any shell snippets shown to the user.
- `.claude/` is untracked at repo root — leave it alone unless the user asks.
