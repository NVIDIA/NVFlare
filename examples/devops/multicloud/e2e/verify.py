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
import re
import shlex
import shutil
import sys
from pathlib import Path

import yaml


def _resolve_path(config_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def participants(args):
    config_path = Path(args.config).resolve()
    kubeconfig_dir = Path(args.kubeconfig_dir).resolve()
    raw = yaml.safe_load(config_path.read_text())
    clouds = raw.get("clouds") or {}
    entries = raw.get("participants") or []
    if not clouds or not entries:
        raise SystemExit(f"{config_path}: missing clouds or participants")

    rows = []
    default_image = None
    for entry in entries:
        cloud = entry["cloud"]
        cloud_cfg = clouds.get(cloud) or {}
        merged = {**cloud_cfg, **entry}
        prepare = entry.get("prepare") or cloud_cfg.get("prepare") or {}
        parent = prepare.get("parent") or {}
        image = parent.get("docker_image")
        if image and default_image is None:
            default_image = image

        raw_kubeconfig = merged.get("kubeconfig")
        kubeconfig = (
            _resolve_path(config_path, raw_kubeconfig)
            if raw_kubeconfig
            else str((kubeconfig_dir / f"{cloud}.yaml").resolve())
        )
        study_data = merged.get("study_data") or {}
        rows.append(
            {
                "role": entry["role"],
                "name": entry["name"],
                "namespace": args.namespace or merged["namespace"],
                "kubeconfig": kubeconfig,
                "cloud": cloud,
                "image": image,
                "has_study_dataset": bool((study_data.get(args.study) or {}).get(args.dataset)),
            }
        )

    servers = [r for r in rows if r["role"] == "server"]
    clients = [r for r in rows if r["role"] != "server"]
    if len(servers) != 1:
        raise SystemExit(f"{config_path}: expected exactly one server participant, found {len(servers)}")
    if len(clients) < args.num_clients:
        raise SystemExit(f"{config_path}: --num-clients={args.num_clients}, but config has only {len(clients)} clients")
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


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def create_numpy_configs(args):
    server_config = {
        "format_version": 2,
        "workflows": [
            {
                "id": "controller",
                "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                "args": {
                    "aggregation_weights": {},
                    "num_clients": args.num_clients,
                    "num_rounds": args.num_rounds,
                },
            }
        ],
        "components": [
            {
                "id": "json_generator",
                "path": "nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator",
                "args": {},
            },
            {
                "id": "model_selector",
                "path": "nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector",
                "args": {"aggregation_weights": {}, "key_metric": "weight_mean"},
            },
            {
                "id": "persistor",
                "path": "nvflare.app_common.np.np_model_persistor.NPModelPersistor",
                "args": {
                    "model_name": "server.npy",
                    "model": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                },
            },
        ],
        "task_result_filters": [],
        "task_data_filters": [],
    }
    client_config = {
        "format_version": 2,
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": "nvflare.app_common.executors.in_process_client_api_executor.InProcessClientAPIExecutor",
                    "args": {
                        "task_script_path": "e2e_numpy_client.py",
                        "task_script_args": "--update_type full",
                    },
                },
            }
        ],
        "task_result_filters": [],
        "task_data_filters": [],
        "components": [],
    }
    return server_config, client_config


def create_cifar10_configs(args):
    server_config = {
        "format_version": 2,
        "model_class_path": "e2e_net.E2ENet",
        "workflows": [
            {
                "id": "fedavg_ctl",
                "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                "args": {
                    "num_clients": args.num_clients,
                    "num_rounds": args.num_rounds,
                    "persistor_id": "persistor",
                },
            }
        ],
        "components": [
            {
                "id": "persistor",
                "path": "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
                "args": {"model": {"path": "e2e_net.E2ENet"}},
            }
        ],
    }

    client_args = [
        "--data-root",
        args.data_root,
        "--num-clients",
        str(args.num_clients),
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

    client_config = {
        "format_version": 2,
        "executors": [
            {
                "tasks": ["train", "validate", "submit_model"],
                "executor": {
                    "path": "nvflare.app_opt.pt.in_process_client_api_executor.PTInProcessClientAPIExecutor",
                    "args": {
                        "task_script_path": "e2e_cifar10_client.py",
                        "task_script_args": " ".join(shlex.quote(arg) for arg in client_args),
                        "params_transfer_type": "FULL",
                        "train_with_evaluation": True,
                        "result_pull_interval": 0.5,
                        "log_pull_interval": 0.1,
                        "train_task_name": "train",
                        "evaluate_task_name": "validate",
                        "submit_model_task_name": "submit_model",
                    },
                },
            }
        ],
        "task_result_filters": [],
        "task_data_filters": [],
        "components": [],
    }
    return server_config, client_config


def create_job(args):
    job_dir = Path(args.job_dir)
    templates_dir = Path(args.templates_dir)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    (job_dir / "app_server" / "config").mkdir(parents=True)
    (job_dir / "app_server" / "custom").mkdir(parents=True)
    (job_dir / "app_client" / "config").mkdir(parents=True)
    (job_dir / "app_client" / "custom").mkdir(parents=True)

    selected_clients = [line.strip() for line in Path(args.selected_clients).read_text().splitlines() if line.strip()]
    participants_data = read_participants(Path(args.participants_tsv))
    launcher_spec = {"default": {"k8s": k8s_spec(args.job_image, args.python_path)}}
    for participant in participants_data:
        launcher_spec[participant["name"]] = {"k8s": k8s_spec(participant["image"], args.python_path)}

    meta = {
        "name": args.job_name,
        "resource_spec": {},
        "min_clients": args.num_clients,
        "deploy_map": {
            "app_server": ["server"],
            "app_client": selected_clients,
        },
        "launcher_spec": launcher_spec,
    }
    write_json(job_dir / "meta.json", meta)

    if args.job_type == "numpy":
        server_config, client_config = create_numpy_configs(args)
        copy_file(
            templates_dir / "numpy" / "app_client" / "custom" / "e2e_numpy_client.py",
            job_dir / "app_client" / "custom" / "e2e_numpy_client.py",
        )
    else:
        server_config, client_config = create_cifar10_configs(args)
        copy_file(
            templates_dir / "cifar10" / "app_server" / "custom" / "e2e_net.py",
            job_dir / "app_server" / "custom" / "e2e_net.py",
        )
        copy_file(
            templates_dir / "cifar10" / "app_server" / "custom" / "e2e_net.py",
            job_dir / "app_client" / "custom" / "e2e_net.py",
        )
        copy_file(
            templates_dir / "cifar10" / "app_client" / "custom" / "e2e_cifar10_client.py",
            job_dir / "app_client" / "custom" / "e2e_cifar10_client.py",
        )

    write_json(job_dir / "app_server" / "config" / "config_fed_server.json", server_config)
    write_json(job_dir / "app_client" / "config" / "config_fed_client.json", client_config)


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
        print(f"{args.namespace}\t{pod_name}\t{restarts}\t{args.participant}")


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
        ns, pod, restarts, participant = line.split("\t")
        data[(ns, pod)] = (int(restarts), participant)
    return data


def compare_restarts(args):
    before = _load_restarts(Path(args.before))
    after = _load_restarts(Path(args.after))
    failures = []
    for key, (after_count, participant) in after.items():
        before_count = before.get(key, (0, participant))[0]
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
    bad = re.search(
        r"(?m)(\bERROR\b|CRITICAL|Traceback \(most recent call last\)|\bException\b|"
        r"ReturnCode\.EXECUTION_EXCEPTION|FINISHED_EXCEPTION)",
        bad_log_text,
    )
    if bad:
        raise SystemExit(f"job logs contain error marker: {bad.group(1)}")

    for round_num in range(args.num_rounds):
        marker = f"E2E_ROUND current_round={round_num}"
        if marker not in log_text:
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
