#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

MULTICLOUD_DIR = Path(__file__).resolve().parents[1]


def _load_deploy_module():
    # Reuse deploy.py's config parsing so the E2E targets exactly what deploy.py deployed.
    if str(MULTICLOUD_DIR) not in sys.path:
        sys.path.insert(0, str(MULTICLOUD_DIR))
    import deploy

    return deploy


def participants(args):
    deploy = _load_deploy_module()
    if args.kubeconfig_dir:
        deploy.DEFAULT_KUBECONFIG_DIR = Path(args.kubeconfig_dir).resolve()
    try:
        config = deploy.load_config(Path(args.config).resolve())
    except ValueError as e:
        raise SystemExit(str(e))

    rows = []
    default_image = None
    for p in config.participants:
        image = (p.prepare.get("parent") or {}).get("docker_image")
        if image and default_image is None:
            default_image = image
        rows.append(
            {
                "role": p.role,
                "name": p.name,
                "namespace": args.namespace or p.namespace,
                "kubeconfig": p.kubeconfig,
                "cloud": p.cloud,
                "image": image,
                "has_study_dataset": bool((p.study_data.get(args.study) or {}).get(args.dataset)),
            }
        )

    servers = [r for r in rows if r["role"] == "server"]
    clients = [r for r in rows if r["role"] != "server"]
    if len(clients) < args.num_clients:
        raise SystemExit(f"{args.config}: --num-clients={args.num_clients}, but config has only {len(clients)} clients")
    if not (args.job_image or default_image):
        raise SystemExit("No job image found in config. Pass --job-image.")

    selected_rows = [servers[0]] + clients[: args.num_clients]
    for row in selected_rows:
        job_image = args.job_image or row["image"] or default_image
        if not job_image:
            raise SystemExit(f"No job image found for participant {row['name']}. Pass --job-image.")
        if (
            row["role"] != "server"
            and args.job_type == "cifar10"
            and not args.allow_download
            and args.data_root == f"/data/{args.study}/{args.dataset}"
            and not row["has_study_dataset"]
        ):
            raise SystemExit(
                f"participant {row['name']} has no study_data mapping for {args.study}/{args.dataset}; "
                "preloaded-data mode will not mount the dataset. Add study_data to the deploy config, "
                "lower --num-clients, pass --data-root for a different mounted path, or use --download for demo runs."
            )
        print("\t".join([row["role"], row["name"], row["namespace"], row["kubeconfig"], row["cloud"], job_image]))


def read_participants(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        role, name, namespace, kubeconfig, cloud, image = line.split("\t")
        rows.append(
            {
                "role": role,
                "name": name,
                "namespace": namespace,
                "kubeconfig": kubeconfig,
                "cloud": cloud,
                "image": image,
            }
        )
    return rows


def k8s_spec(image: str, python_path: str) -> dict:
    return {
        "image": image,
        "python_path": python_path,
        "num_of_gpus": 0,
        "cpu": "1",
        "memory": "2Gi",
        "ephemeral_storage": "4Gi",
    }


def _create_numpy_recipe(args):
    from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
    from nvflare.client.config import TransferType

    return NumpyFedAvgRecipe(
        name=args.job_name,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        train_script="e2e_numpy_client.py",
        train_args="--update_type full",
        launch_external_process=False,
        params_transfer_type=TransferType.FULL,
        key_metric="weight_mean",
    )


def _create_cifar10_recipe(args, selected_clients, model_dir: Path):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import TransferType

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from e2e_net import E2ENet

    client_args = [
        "--data-root",
        args.data_root,
        "--sites",
        ",".join(selected_clients),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-val-samples",
        str(args.max_val_samples),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--torch-threads",
        str(args.torch_threads),
    ]
    if args.allow_download:
        client_args.append("--download")
    for arg in client_args:
        # The in-process script runner splits task_script_args on whitespace with
        # no unquoting, so values containing whitespace cannot be passed through.
        if any(c.isspace() for c in arg):
            raise SystemExit(f"job script argument {arg!r} contains whitespace, which cannot be passed to the client")

    return FedAvgRecipe(
        name=args.job_name,
        model=E2ENet(),
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        train_script="e2e_cifar10_client.py",
        train_args=" ".join(client_args),
        launch_external_process=False,
        params_transfer_type=TransferType.FULL,
        key_metric="accuracy",
    )


def _strip_metrics_artifact_writer(job_dir: Path):
    # BaseFedJob injects MetricsArtifactWriter with no opt-out; deployed FLARE
    # parents may run an older NVFlare without that module, which aborts the job
    # at config load. The smoke checks do not use metrics artifacts, so drop it.
    for config_path in job_dir.rglob("config_fed_server.json"):
        config = json.loads(config_path.read_text())
        components = config.get("components") or []
        filtered = [c for c in components if c.get("id") != "metrics_artifact_writer"]
        if len(filtered) != len(components):
            config["components"] = filtered
            config_path.write_text(json.dumps(config, indent=4) + "\n")


def _patch_meta(job_dir: Path, args, selected_clients, participants_data):
    launcher_spec = {"default": {"k8s": k8s_spec(args.job_image, args.python_path)}}
    for participant in participants_data:
        launcher_spec[participant["name"]] = {"k8s": k8s_spec(participant["image"], args.python_path)}

    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["launcher_spec"] = launcher_spec
    # The exported deploy_map targets @ALL, which would also deploy to connected
    # clients outside this run's selection; pin it to the selected clients.
    for app, targets in (meta.get("deploy_map") or {}).items():
        if "@ALL" in targets:
            meta["deploy_map"][app] = ["server"] + selected_clients
    meta_path.write_text(json.dumps(meta, indent=4) + "\n")


def create_job(args):
    try:
        import nvflare  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"create-job needs the nvflare package in the helper python ({sys.executable}); "
            f"set PYTHON_BIN to a python with NVFlare installed: {e}"
        )
    if args.job_type == "cifar10":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                f"create-job --job-type cifar10 needs torch in the helper python ({sys.executable}) "
                f"to build the initial model; install torch or use --job-type numpy: {e}"
            )

    if not args.job_name or os.sep in args.job_name or args.job_name in (".", ".."):
        raise SystemExit(f"--job-name must be a simple directory name, got {args.job_name!r}")

    job_dir = Path(args.job_dir).resolve()
    templates_dir = Path(args.templates_dir).resolve()
    selected_clients = [line.strip() for line in Path(args.selected_clients).read_text().splitlines() if line.strip()]
    participants_data = read_participants(Path(args.participants_tsv).resolve())

    if job_dir.exists():
        shutil.rmtree(job_dir)
    # Export into a staging dir so the recipe's <job_name> output cannot collide
    # with unrelated entries next to --job-dir.
    staging_dir = job_dir.parent / f"{job_dir.name}.export"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # The recipe resolves train_script against the working directory and copies it
    # into the job's custom dir, so export from the template script's directory.
    old_cwd = os.getcwd()
    try:
        if args.job_type == "numpy":
            os.chdir(templates_dir / "numpy" / "app_client" / "custom")
            recipe = _create_numpy_recipe(args)
        else:
            os.chdir(templates_dir / "cifar10" / "app_client" / "custom")
            recipe = _create_cifar10_recipe(args, selected_clients, templates_dir / "cifar10" / "app_server" / "custom")
        recipe.export(str(staging_dir))
    finally:
        os.chdir(old_cwd)

    (staging_dir / args.job_name).rename(job_dir)
    shutil.rmtree(staging_dir)
    _strip_metrics_artifact_writer(job_dir)
    _patch_meta(job_dir, args, selected_clients, participants_data)


def json_get(args):
    data = json.load(sys.stdin)
    cur = data
    for part in args.expr.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (IndexError, ValueError):
                cur = None
        else:
            cur = None
        if cur is None:
            break
    print("" if cur is None else cur)


def pod_restarts(args):
    payload = json.load(sys.stdin)
    for pod in payload.get("items", []):
        pod_name = pod.get("metadata", {}).get("name", "")
        statuses = pod.get("status", {}).get("containerStatuses") or []
        restarts = sum(int(s.get("restartCount", 0) or 0) for s in statuses)
        # Pods already Terminating or in a terminal phase are expected to
        # disappear; mark them so compare-restarts does not flag their deletion.
        terminating = bool(pod.get("metadata", {}).get("deletionTimestamp"))
        terminal_phase = pod.get("status", {}).get("phase") in ("Succeeded", "Failed")
        expect_present = "0" if terminating or terminal_phase else "1"
        print(f"{args.namespace}\t{pod_name}\t{restarts}\t{args.participant}\t{expect_present}")


def list_job_pods(args):
    payload = json.load(sys.stdin)
    prefixes = (args.job_id, f"j{args.job_id}")
    for pod in payload.get("items", []):
        name = pod.get("metadata", {}).get("name", "")
        if name.startswith(prefixes):
            print(name)


def _load_restarts(path: Path):
    data = {}
    for line in path.read_text().splitlines():
        ns, pod, restarts, participant, expect_present = line.split("\t")
        data[(ns, pod)] = (int(restarts), participant, expect_present == "1")
    return data


def compare_restarts(args):
    before = _load_restarts(Path(args.before))
    after = _load_restarts(Path(args.after))
    failures = []
    # A stable pre-existing pod that disappears was deleted mid-run; a crashed
    # parent replaced by its Deployment shows up only this way (the replacement
    # pod has a new name and restartCount=0). Pods that were already Terminating
    # or Completed in the before snapshot are expected to go away.
    for key, (_count, _participant, expect_present) in before.items():
        if expect_present and key not in after:
            failures.append(f"{key[0]}/{key[1]} existed before the run but disappeared")
    for key, (after_count, participant, _expect_present) in after.items():
        before_count = before.get(key, (0, participant, True))[0]
        if key in before and after_count > before_count:
            failures.append(f"{key[0]}/{key[1]} restarted from {before_count} to {after_count}")
        elif key not in before and after_count > 0:
            failures.append(f"{key[0]}/{key[1]} started with restartCount={after_count}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


def validate_result(args):
    download = json.loads(Path(args.download_json).read_text())
    logs = json.loads(Path(args.logs_json).read_text())
    data = download.get("data") or {}
    artifacts = data.get("artifacts") or {}
    download_path = data.get("download_path") or data.get("path")

    if "global_model" not in artifacts:
        root = Path(download_path) if download_path else None
        names = {
            "FL_global_model.pt",
            "global_model.pt",
            "global_model.pth",
            "best_FL_global_model.pt",
            "best_global_model.pt",
            "model_global.json",
            "model_global.joblib",
            "server.npy",
            "model.npy",
            "last_global_model.npy",
            "best_global_model.npy",
        }
        found = []
        if root and root.exists():
            found = [
                p
                for p in root.rglob("*")
                if p.is_file() and (p.name in names or (args.job_type == "numpy" and p.suffix == ".npy"))
            ]
        if not found:
            raise SystemExit("downloaded results do not contain a global model artifact")

    cli_logs = (logs.get("data") or {}).get("logs") or {}
    log_texts = [
        "\n".join(value if isinstance(value, str) else json.dumps(value, sort_keys=True) for value in cli_logs.values())
    ]
    if download_path:
        root = Path(download_path)
        if root.exists():
            log_texts.append("\n".join(p.read_text(errors="ignore") for p in root.rglob("log.txt")))
    if args.k8s_logs:
        k8s_log_path = Path(args.k8s_logs)
        if k8s_log_path.exists():
            log_texts.append(k8s_log_path.read_text(errors="ignore"))
    log_text = "\n".join(t for t in log_texts if t)
    if not log_text:
        raise SystemExit("no job logs found in CLI output, downloaded result, or Kubernetes pod logs")

    benign_log_noise = (
        "_LogTailProducer",
        "no stream processing info registered for log_streaming:live_log",
    )
    bad_log_text = "\n".join(
        line for line in log_text.splitlines() if not any(token in line for token in benign_log_noise)
    )
    # EXECUTION_EXCEPTION matches bare, so it also covers the FINISHED:EXECUTION_EXCEPTION
    # and ReturnCode.EXECUTION_EXCEPTION spellings that appear in job logs.
    fatal = re.search(
        r"(?m)(Traceback \(most recent call last\)|EXECUTION_EXCEPTION|FINISHED_EXCEPTION)",
        bad_log_text,
    )
    if fatal:
        raise SystemExit(f"job logs contain error marker: {fatal.group(1)}")
    # Error-level lines are routine during job-pod shutdown (peer connections closing),
    # and the structured terminal job status is checked before this runs; warn only.
    warning = re.search(r"(?m)(\bERROR\b|CRITICAL|\bException\b)", bad_log_text)
    if warning:
        print(
            f"WARNING: job logs contain '{warning.group(1)}' lines; "
            "the job finished with a completed status, so this is not treated as a failure",
            file=sys.stderr,
        )

    for round_num in range(args.num_rounds):
        marker = f"E2E_ROUND current_round={round_num}"
        # \b keeps round 1 from matching round 10, 11, ...
        if not re.search(re.escape(marker) + r"\b", log_text):
            raise SystemExit(f"missing round log marker: {marker}")


def build_parser():
    parser = argparse.ArgumentParser(description="Helpers for multicloud Kubernetes E2E verification.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("participants")
    p.add_argument("--config", required=True)
    p.add_argument("--kubeconfig-dir", required=True)
    p.add_argument("--namespace", default="")
    p.add_argument("--num-clients", type=int, required=True)
    p.add_argument("--job-image", default="")
    p.add_argument("--job-type", choices=["cifar10", "numpy"], required=True)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--study", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.set_defaults(func=participants)

    p = sub.add_parser("create-job")
    p.add_argument("--job-dir", required=True)
    p.add_argument("--job-name", required=True)
    p.add_argument("--job-type", choices=["cifar10", "numpy"], required=True)
    p.add_argument("--job-image", required=True)
    p.add_argument("--python-path", required=True)
    p.add_argument("--num-rounds", type=int, required=True)
    p.add_argument("--num-clients", type=int, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--max-train-samples", type=int, required=True)
    p.add_argument("--max-val-samples", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--torch-threads", type=int, required=True)
    p.add_argument("--selected-clients", required=True)
    p.add_argument("--participants-tsv", required=True)
    p.add_argument("--templates-dir", required=True)
    p.set_defaults(func=create_job)

    p = sub.add_parser("json-get")
    p.add_argument("expr")
    p.set_defaults(func=json_get)

    p = sub.add_parser("pod-restarts")
    p.add_argument("--participant", required=True)
    p.add_argument("--namespace", required=True)
    p.set_defaults(func=pod_restarts)

    p = sub.add_parser("list-job-pods")
    p.add_argument("--job-id", required=True)
    p.set_defaults(func=list_job_pods)

    p = sub.add_parser("compare-restarts")
    p.add_argument("before")
    p.add_argument("after")
    p.set_defaults(func=compare_restarts)

    p = sub.add_parser("validate-result")
    p.add_argument("--download-json", required=True)
    p.add_argument("--logs-json", required=True)
    p.add_argument("--k8s-logs", default="")
    p.add_argument("--num-rounds", type=int, required=True)
    p.add_argument("--job-type", choices=["cifar10", "numpy"], required=True)
    p.set_defaults(func=validate_result)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
