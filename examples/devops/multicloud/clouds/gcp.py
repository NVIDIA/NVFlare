from __future__ import annotations

import yaml

from .base import CloudProvider


class GcpProvider(CloudProvider):
    name = "gcp"
    auth_check_cmd = ["gcloud", "auth", "print-access-token"]
    auth_failed_message = "GCP auth failed. Run: gcloud auth login"
    auth_expired_message = "GCP auth expired. Run: gcloud auth login"

    def _resolve_project(self, run, project: str | None) -> str:
        if isinstance(project, str) and project.strip():
            return project.strip()

        r = run(["gcloud", "config", "get-value", "project"], capture=True, check=False)
        project = r.stdout.strip()
        if r.returncode == 0 and project and project != "(unset)":
            return project

        raise RuntimeError("GCP project is required. Run 'gcloud config set project <project>'.")

    def parse_kubeconfig(self, kc_path):
        data = yaml.safe_load(kc_path.read_text())
        current_ctx = data.get("current-context")
        if not current_ctx:
            raise ValueError(f"{kc_path}: no current-context")
        ctx = next((c for c in data.get("contexts", []) if c.get("name") == current_ctx), None)
        if not ctx:
            raise ValueError(f"{kc_path}: current-context {current_ctx!r} not found in contexts")
        cluster_name = ctx["context"]["cluster"]
        parts = cluster_name.split("_")
        if len(parts) < 4 or parts[0] != "gke":
            raise ValueError(
                f"{kc_path}: GKE context cluster {cluster_name!r} not in 'gke_<project>_<region>_<cluster>' form"
            )
        return {"project": parts[1], "region": parts[2]}

    def reserve_ip(self, *, run, ip_tag, gcp_project=None, gcp_region=None, **kwargs):
        gcp_project = self._resolve_project(run, gcp_project)
        gcp_region = gcp_region or "us-central1"
        print(f"Ensuring static IP {ip_tag} ...")
        run(
            [
                "gcloud",
                "compute",
                "addresses",
                "create",
                ip_tag,
                f"--region={gcp_region}",
                f"--project={gcp_project}",
                "--quiet",
            ],
            check=False,
        )
        r = run(
            [
                "gcloud",
                "compute",
                "addresses",
                "describe",
                ip_tag,
                f"--region={gcp_region}",
                f"--project={gcp_project}",
                "--format=value(address)",
            ],
            capture=True,
        )
        ip = r.stdout.strip()
        print(f"  Using: {ip} ({ip_tag})")
        return ip, ip_tag

    def release_ip(self, *, run, ip_name, state):
        gcp_project = self._resolve_project(run, state.get("gcp_project"))
        gcp_region = state.get("gcp_region") or "us-central1"
        print(f"Releasing IP {ip_name} ...")
        run(
            [
                "gcloud",
                "compute",
                "addresses",
                "delete",
                ip_name,
                f"--region={gcp_region}",
                f"--project={gcp_project}",
                "--quiet",
            ],
            check=False,
        )

    def server_service_helm_args(self, *, server_ip, state):
        return ["--set", f"service.loadBalancerIP={server_ip}"]
