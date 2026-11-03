# AGENTS.md

This repository is worked on by several AI coding tools (Claude Code, Google Antigravity, others).
All tools follow the same rules:

1. Read `CLAUDE.md` (working rules, commands, definition of done) and `SPEC.md` (the specification) before doing anything.
2. Stay inside the files assigned to you in the task; do not change contracts in SPEC.md silently.
3. Never invent results or metrics; run the code.
4. After each task: tests + lint green, append to `docs/LEARNING_LOG.md`, add to `DECISIONS.md` if you chose between alternatives.
5. Before ending a session, write a short HANDOFF note at the end of `docs/LEARNING_LOG.md`:
   what is done, what is half-done (files + function names), what to do next, how to run the tests.
   The next tool (possibly a different one) starts from that note.
