# Agent Orchestrator — Project Brief

Best-of-both-worlds brief merging the Sonnet 4.6 + Opus 4.6 drafts. Hand
this to the GUI architect (Sonnet/Opus) to kick off a build session.

## What We're Building

A two-tier AI workflow:

- **GUI (Claude Sonnet/Opus) = Architect** — plans tasks, generates configs
  and prompts, reviews diffs interactively.
- **CLI agents (Claude Haiku, escalatable) = Workers** — execute scoped
  tasks in parallel on separate git branches and open PRs.

## Repo Context

- `UQ4ML_WaterTemp` — Python + Keras ML pipeline for water-temperature
  uncertainty quantification.
- Primary deliverable: trained `.keras` model files.
- Candidate parallel tracks: refactoring, documentation, performance
  optimization, ML training sweeps.

## Workflow

1. I describe tasks in the GUI chat.
2. GUI generates `tasks.yaml`, `prompts/*.md`, and reviews `orchestrator.py`.
3. I run `python orchestrator/orchestrator.py` in my terminal.
4. Agents work in parallel, each on its own branch, committing as they go.
5. Each agent opens a GitHub PR with a summary.
6. I return to the GUI: "review the agent results" — GUI walks me through
   each diff and helps me decide what to merge.

## Stack

- Anthropic API key via `ANTHROPIC_API_KEY` env var (not Claude Pro).
- Agent runner: **`claude` CLI** invoked via `subprocess` (not direct SDK
  calls) — this gets us tool use, file editing, and git for free.
- Orchestration: Python `asyncio` + `subprocess`, stdlib + `pyyaml` only.
- PRs: `gh` CLI.
- Logs: JSONL per agent under `agent_logs/<timestamp>/<task>.jsonl`.

## Model Strategy

- Default workers: `claude-haiku-4-5-20251001` (cheap, fast).
- Escalate to `claude-sonnet-4-6` for ML training / complex refactors.
- `model` is a **swappable string field** per task — keep the abstraction
  loose enough to swap in `openai`/`ollama` runners later.

## File Layout

```
orchestrator/
├── orchestrator.py       # async launcher (subprocess → `claude` CLI)
├── tasks.yaml            # task definitions
├── prompts/              # one .md per task
│   ├── example_docs.md
│   └── example_optimize.md
├── agent_logs/           # JSONL logs, gitignored
├── BRIEF.md              # this file
└── README.md             # how to run
```

## Task Schema (`tasks.yaml`)

Each entry:

- `name` — short slug
- `branch` — git branch (e.g. `agent/docs-refactor`)
- `model` — model ID string
- `max_turns` — per-agent budget
- `prompt_file` — path to scoped prompt
- `file_scope` — explicit allowlist of files the agent may modify

## Prompt Requirements

Every `prompts/*.md` must declare:

- **Files in scope** (only modify these).
- **Reference-only files** (read, don't modify).
- **Goals + acceptance criteria.**
- **Constraints** — no signature changes, no new deps unless approved,
  conventional commits, one commit per logical change.

## Constraints (non-negotiable)

- Non-overlapping file scopes across agents to prevent merge conflicts.
- Stdlib + `pyyaml` only — no heavy frameworks.
- Cross-platform (Windows/Mac/Linux). Use `pathlib`, avoid shell-only ops.
- Per-agent timeout + token budget; errors logged, never silently swallowed.
- Final `REPORT.md` summarizing each agent's outcome (success / failure /
  files touched / PR URL).

## Architect: How To Start

1. Read the repo tree to understand current structure (driver/, evaluations/,
   helper/, etc.).
2. Ask me clarifying questions about the ML training tracks before writing
   real task configs.
3. Generate the files above tailored to actual scripts in the repo.
4. Keep it simple — don't over-engineer.
