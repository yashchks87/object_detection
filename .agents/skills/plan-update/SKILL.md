---
name: plan-update
description: Summarize the current chat into a compact, complete session log under plans/ so work can be resumed after the repo is re-cloned or the cluster is wiped
argument-hint: "[optional topic / file name hint]"
triggers:
  - user
allowed-tools:
  - read
  - grep
  - glob
permissions:
  allow:
    - Write(plans/**)
    - Exec(git status)
    - Exec(git log)
    - Exec(git diff)
    - Exec(ls)
---

Write (or update) a session log in `plans/` that captures everything done in
THIS chat, so anyone can pick the work back up from the plan file alone after
the repo is re-cloned or the cluster is wiped. The chat is the source; the
repository and run state are used to verify facts.

## 1. Decide the target file

- Read the existing files in `plans/` first (at least their headers) to match
  their style and to avoid duplicates.
- If a session log for this same chat/topic already exists (e.g. written
  earlier in this chat), UPDATE it in place: append new numbered sections and
  refresh the final "state" section. Do not create a second file.
- Otherwise create `plans/YYYY-MM-DD-<topic-slug>-session.md` (today's date,
  short kebab-case topic, e.g. `2026-09-25-frcnn-training-session.md`). If the
  user passed an argument, use it as the topic/file-name hint.
- Experiment notes (`plans/expNNN-<slug>.md`) are separate and only written
  when the user asks for one.

## 2. Gather and verify facts (do not rely on memory alone)

- `git status`, `git log --oneline -10`: what is committed, what is not.
- Check the files the chat created/changed exist, and re-read key values
  (defaults, paths, versions) from the code rather than recalling them.
- If jobs were launched: check their log tail / PID files for the current
  state (running, finished, failed, last epoch/step, metrics).
- Anything that could not be verified is written as unverified. Never invent
  results, numbers, or outcomes.

## 3. Document structure

Use this skeleton (shown as plain text; the output file is Markdown):

````text
# Session log: <topic>

Date: YYYY-MM-DD
Chat transcript (condensed but complete) between <user> and Devin covering
<one-sentence scope>.

---

## 1. <First request, short title>

**Request:** <what the user asked, in one or two sentences>

<Findings / environment discovered / diagnosis, as short bullets.>
<Decisions made and WHY; options considered and rejected and why.>
<Files created or changed (table: File | Purpose) when there are several.>
<Verification: tests run, measured numbers (tables for benchmarks), outcomes.>

## 2. <Next request / incident / question> ...
(one numbered section per request, question, incident, or decision, in chat order)

---

## Current state (end of session)

- What is running / finished (run names, PIDs, log paths, output paths).
- Exact commands to launch / monitor / stop / resume (fenced bash block).
- Uncommitted changes, leftovers that can be deleted.
- Open items, unverified things, and agreed next steps.
````

## 4. Writing rules

- Compact but complete: condensed prose and bullets, tables for comparisons,
  benchmarks, file lists, and schemas. No filler, no restating the obvious.
- Include everything needed to reproduce or resume: absolute paths, exact
  commands and flags, env vars, pinned versions, hardware, dataset
  sizes/splits, default hyperparameters, measured numbers with units.
- Record the reasoning, not just the outcome: why a design was chosen, what
  went wrong, root cause, and the fix. Keep bugs/incidents and their lessons.
- Record the user's decisions, preferences, and rejections (e.g. "Yash chose
  to run the launch himself"), and questions the user asked with the answers.
- Only small code/config excerpts when they are essential (a command, a
  schema line); never paste whole files.
- Never write secrets: no tokens, API keys, passwords, or netrc contents.
  Mention only that credentials exist and where they are configured.
- Chronological order; use the same headings/tone as the existing plans.

## 5. Finish

- Do not commit or push unless the user asks.
- Reply briefly: the file path (as a clickable reference), whether it was
  created or updated, and a 3-6 bullet outline of what it covers, plus
  anything flagged as unverified.
