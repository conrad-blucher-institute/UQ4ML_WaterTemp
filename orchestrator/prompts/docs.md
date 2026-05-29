# Agent task: documentation pass (DUMMY)

This is a placeholder prompt used to smoke-test the orchestrator. Replace the
"Files in scope" and "Goal" sections with real targets before running for real.

## Goal
Create a single file `orchestrator/agent_output/dummy_docs.md` containing a
two-paragraph summary of what a real documentation agent would do in this repo.

## Files in scope (ONLY create/modify these)
- orchestrator/agent_output/dummy_docs.md

## Files for reference only (read, do not modify)
- README.md (if present)

## Constraints
- Do not modify any file outside the scope list.
- Do not install dependencies.
- Do not run long commands.
- Make exactly one git commit when finished, message: `docs(agent): dummy docs pass`.

## Done criteria
- The output file exists and is non-empty.
- One commit on the current branch.
