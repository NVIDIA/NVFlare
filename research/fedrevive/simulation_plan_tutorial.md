# Compare and customize simulation plans

This small example generates two plans without a dataset or training run. Six
logical clients compete for three active assignments. Each accepted upload
publishes a model, and the run stops after 12 publications. The second plan
slows only `client-0` uploads. All times are **virtual time units**; they are
not measured training durations.

Save the block below as `plan_demo.py` and run `python plan_demo.py` from
`research/fedrevive` in an environment with NVFlare, NumPy, and Matplotlib
installed. It writes validated JSON plans and PNG plots to `plan_demo/`.

```python
import json
from pathlib import Path

from nvflare.apis.simulation_plan import validate_simulation_plan
from plot_plan import plot_simulation_plan
from schedule_standard import make_simulation_plan, sample_paper_delay

profiles = {
    f"client-{i}": {
        "download_mean": 0.10,
        "train_mean": 0.70,
        "upload_mean": 0.15,
        "upload_half_width": 0.02,
    }
    for i in range(6)
}
settings = dict(
    run_seed=10,
    max_time=20.0,
    num_rounds=12,
    concurrency=3,
    min_open_slots=1,
    buffer_size=1,
)


def baseline_delay(assignment_id, logical_name, phase, profile, rng):
    return sample_paper_delay(phase, profile, rng)


def slow_upload(assignment_id, logical_name, phase, profile, rng):
    duration = baseline_delay(assignment_id, logical_name, phase, profile, rng)
    return duration * 20 if logical_name == "client-0" and phase == "upload" else duration


output = Path("plan_demo")
output.mkdir(exist_ok=True)
plans = {}
for label, sampler in (("baseline", baseline_delay), ("slow_upload", slow_upload)):
    plan = validate_simulation_plan(make_simulation_plan(profiles, **settings, delay_sampler=sampler))
    (output / f"{label}.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    figure = plot_simulation_plan(plan, limit=14)
    figure.savefig(output / f"{label}.png", dpi=150)
    plans[label] = plan


def first_client_zero_upload(plan):
    execution = plan["async_execution"]
    uploads = [
        event
        for event in execution["timing"]["events"]
        if event["phase"] == "upload" and event["virtual_time"] < execution["timing"]["horizon"]
    ]
    for accepted_before, event in enumerate(uploads):
        assignment_id = event["assignment_id"]
        if execution["logical_sequence"][assignment_id] == "client-0":
            return accepted_before + 1, accepted_before - execution["base_versions"][assignment_id]
    raise ValueError("client-0 has no accepted upload in this plan")


for label, plan in plans.items():
    rank, staleness = first_client_zero_upload(plan)
    print(f"{label}: first client-0 upload accepted #{rank}, staleness {staleness}")
```

Both plans use the custom sampler path, so unchanged assignment/phase samples
use the same seed scheme. Changing one duration can change later client
selection and model bases. With `buffer_size=1`, staleness at acceptance is
the number of earlier accepted uploads minus the assignment's base version.
For larger buffers, count **published versions** instead of uploads.
With these settings, the printed acceptance position and staleness change from
`#1, 0` to `#7, 6` for the first `client-0` upload.

Open the two PNGs to compare phase lengths and publication timing. The
`plot_simulation_plan()` viewer also accepts a saved JSON path, an assignment
window via `first_assignment` and `limit`, and optional serialized payload
sizes for estimating peak temporary storage. See [the viewer options](README.md#visualize-a-plan-before-running-it).

To replay either JSON plan, pass its path to `SimEnv(reprod=True, plan=...)`
with a `FedBuffRecipe` configured for the same 12 publications, concurrency 3,
buffer size 1, and refill threshold 1. The plan names logical clients; the
`SimEnv` physical worker count may be smaller. SimEnv preflights the plan
before training and verifies the recorded logical events afterward. See
[FedRevive replay setup](README.md#replay-a-simulator-plan) and the
[simulation plan guide](../../docs/user_guide/data_scientist_guide/reproducible_simulation.rst)
for the runner command and trace files.
