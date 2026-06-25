# NVFLARE Auto-FL Skill

This skill is intended to optimize any existing NVFLARE `job.py`. It is not
tied to the `research/auto-fl-research` bundle.

## H100 Skill Trial Launcher

For repeatable H100 validation, use
`scripts/launch_h100_skill_trial.sh`. The launcher prepares the environment
before the fresh coding agent starts: it clones the requested NVFLARE branch,
creates a Python 3.12 venv under the trial output directory, installs
job-local requirements when present, removes released `nvflare` packages,
installs the cloned repo editable, installs this skill into an isolated
`CODEX_HOME`, and starts Codex in `tmux` from the selected job directory.

Default lightweight fixture:

```bash
cd /scratch/hroth/Code/nvflare/<checkout>
AUTOFL_H100_BRANCH=codex/autofl-skill-v1 \
  skills/nvflare-autofl/scripts/launch_h100_skill_trial.sh
```

That defaults to:

```text
AUTOFL_H100_JOB=examples/hello-world/hello-pt/job.py
Prompt: Optimize ./job.py for accuracy in sim.
```

Run the bounded 10-candidate product UX smoke test:

```bash
AUTOFL_H100_BRANCH=<branch> \
AUTOFL_H100_JOB=examples/hello-world/hello-pt/job.py \
AUTOFL_H100_PROMPT="Optimize ./job.py for accuracy in sim with a 10-candidate budget." \
AUTOFL_H100_KILL_OLD=1 \
  skills/nvflare-autofl/scripts/launch_h100_skill_trial.sh
```

Run the uncapped product UX trial; this should continue until you interrupt the
tmux session or stop the runner:

```bash
AUTOFL_H100_BRANCH=<branch> \
AUTOFL_H100_JOB=examples/hello-world/hello-pt/job.py \
AUTOFL_H100_PROMPT="Optimize ./job.py for accuracy in sim." \
AUTOFL_H100_KILL_OLD=1 \
  skills/nvflare-autofl/scripts/launch_h100_skill_trial.sh
```

Run the same skill trial on any job in the cloned branch:

```bash
AUTOFL_H100_BRANCH=<branch> \
AUTOFL_H100_JOB=examples/advanced/sklearn-linear/job.py \
AUTOFL_H100_REQUIREMENTS=auto \
  skills/nvflare-autofl/scripts/launch_h100_skill_trial.sh
```

Run the heavier CIFAR research-style fixture only when that is the explicit
test target:

```bash
AUTOFL_H100_BRANCH=<branch> \
AUTOFL_H100_JOB=research/auto-fl-research/tasks/cifar10/job.py \
AUTOFL_H100_PROMPT="Optimize ./job.py for accuracy in sim." \
  skills/nvflare-autofl/scripts/launch_h100_skill_trial.sh
```

When a task-local `mutation_schema.yaml` is present, the deterministic runner
uses its comparison budget automatically. For the CIFAR-10 fixture this means
the H100 profile runs real CIFAR-10 with 8 clients, 20 rounds, 4 local epochs,
cross-site final evaluation, and the profile's timeout rather than the tiny
hello-pt synthetic smoke budget.

Useful overrides:

```bash
AUTOFL_H100_REPO_URL=git@github.com:<user>/NVFlare.git
AUTOFL_H100_JOB=/absolute/path/to/job.py
AUTOFL_H100_JOB_CWD=/absolute/path/to/job-dir
AUTOFL_H100_REQUIREMENTS=/absolute/path/to/requirements.txt
AUTOFL_H100_REQUIREMENTS=none
AUTOFL_H100_NVFLARE_EXTRA=PT
AUTOFL_H100_PROMPT="Optimize ./job.py for AUC in prod with a 10-candidate budget."
AUTOFL_H100_KILL_OLD=1
AUTOFL_H100_BOOTSTRAP_PYTHON=/path/to/python3.12
```

Monitor without interrupting the agent:

```bash
source "$(ls -td /scratch/hroth/Code/nvflare/pr4780-autofl-output/skill_trial_*/session.env | head -1)"
tmux capture-pane -pt "$SESSION" -S -120
tail -f "$OUT/codex-tui.log"
```
