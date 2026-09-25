---
name: setup-env
description: Set up the Python environment for object_detection (apt update, pyenv Python 3.12, pip install requirements) in a fixed order
triggers:
  - user
permissions:
  allow:
    - Exec(sudo apt-get update)
    - Exec(pyenv install)
    - Exec(pyenv local 3.12)
    - Exec(pip install -r requirements.txt)
---

Run these commands in EXACTLY this order, one at a time, starting from `/root`:

1. `sudo apt-get update`
2. `pyenv install 3.12`
3. `pyenv local 3.12`
4. `cd object_detection/`
5. `pip install -r requirements.txt`

## How to run them

- Use ONE persistent shell session (reuse the same `shell_id`) so that the
  `cd` in step 4 and the pyenv version from step 3 carry over to step 5.
  Start that session in `/root`.
- Run each command separately and check its exit code before moving on.
  If a command fails, STOP: do not run the remaining steps. Report the failed
  command and the relevant error output.
- `sudo apt-get update`: if sudo asks for a password, stop and ask the user to
  run it themselves (passwords cannot be entered).
- `pyenv install 3.12`: if Python 3.12 is already installed, pyenv asks an
  interactive "continue? (y/N)" question. Answer `N` (keep the existing
  install) and continue with step 3; this counts as success.
- The installs can take several minutes; wait for them to finish rather than
  starting the next step early.
- Do not add, remove, or reorder steps, and do not edit `requirements.txt`.

## Finish

Report briefly:
- each step with OK / FAILED (and the skip note for step 2 if 3.12 existed),
- `python --version` and `which python` / `which pip` from the same session,
- anything that needs the user's attention.
