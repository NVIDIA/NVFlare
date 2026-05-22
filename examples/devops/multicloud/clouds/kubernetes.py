from __future__ import annotations

import ipaddress
import json

from .base import CloudProvider, service_annotation_args


class KubernetesProvider(CloudProvider):
    name = "kubernetes"
    auth_check_cmd = []
    auth_failed_message = ""
    auth_expired_message = ""

    def _server_config(self, config) -> dict:
        if not config:
            return {}
        return (config.cloud_configs.get(config.server_cloud) or {}).get("server") or {}

    def _server_participant(self, config):
        return next(p for p in config.participants if p.role == "server")

    def _default_server_address(self, config) -> str:
        server = self._server_participant(config)
        return f"nvflare-server.{server.namespace}.svc.cluster.local"

    def _public_service_config(self, config) -> dict:
        return self._server_config(config).get("public_service") or {}

    def _public_service_namespace(self, config, public_service: dict) -> str:
        return public_service.get("namespace") or self._server_participant(config).namespace

    def _public_service_name(self, public_service: dict) -> str:
        return public_service.get("name") or "nvflare-server-public"

    def _run_text(self, run, cmd: list[str], *, check=True) -> str:
        result = run(cmd, check=check, capture=True)
        return (result.stdout or "").strip()

    def validate_server_config(self, config):
        server_config = self._server_config(config)
        service_type = server_config.get("service_type", "ClusterIP")
        if service_type not in {"ClusterIP", "LoadBalancer", "NodePort"}:
            raise SystemExit(f"clouds.{self.name}.server.service_type must be ClusterIP, LoadBalancer, or NodePort")
        public_service = self._public_service_config(config)
        if not public_service:
            return
        metallb = public_service.get("metallb") or {}
        if public_service.get("load_balancer_ip") or metallb.get("pool"):
            return
        raise SystemExit(
            f"clouds.{self.name}.server.public_service requires load_balancer_ip or metallb.pool when enabled"
        )

    def reserve_ip(self, *, run, ip_tag, config=None, **kwargs):
        server_config = self._server_config(config)
        public_service = self._public_service_config(config)
        address = server_config.get("address")
        if not address and public_service:
            address = self._public_service_ip(run, config, public_service)
        if not address:
            address = self._default_server_address(config)
        print(f"Using Kubernetes service address {address} ({self.name})")
        return address, ip_tag

    def _public_service_ip(self, run, config, public_service: dict) -> str:
        configured_ip = public_service.get("load_balancer_ip")
        if configured_ip:
            return configured_ip

        server = self._server_participant(config)
        namespace = self._public_service_namespace(config, public_service)
        name = self._public_service_name(public_service)
        existing_ip = self._run_text(
            run,
            [
                "kubectl",
                "--kubeconfig",
                server.kubeconfig,
                "-n",
                namespace,
                "get",
                "svc",
                name,
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].ip}",
            ],
            check=False,
        )
        if existing_ip:
            return existing_ip

        metallb = public_service.get("metallb") or {}
        pool = metallb.get("pool")
        if not pool:
            return ""
        metallb_namespace = metallb.get("namespace") or "metallb-system"
        pool_range = self._run_text(
            run,
            [
                "kubectl",
                "--kubeconfig",
                server.kubeconfig,
                "-n",
                metallb_namespace,
                "get",
                "ipaddresspool",
                pool,
                "-o",
                "jsonpath={.spec.addresses[0]}",
            ],
        )
        if not pool_range:
            # Dry-run command output is intentionally synthetic.
            return "<metallb-ip>"
        used_ips = self._run_text(
            run,
            [
                "kubectl",
                "--kubeconfig",
                server.kubeconfig,
                "get",
                "svc",
                "-A",
                "-o",
                'jsonpath={range .items[*]}{range .status.loadBalancer.ingress[*]}{.ip}{"\\n"}{end}{end}',
            ],
        )
        used = {line.strip() for line in used_ips.splitlines() if line.strip()}
        return self._first_free_ip(pool_range, used)

    def _first_free_ip(self, pool_range: str, used: set[str]) -> str:
        if "-" in pool_range:
            start_s, end_s = pool_range.split("-", 1)
            start = ipaddress.ip_address(start_s.strip())
            end = ipaddress.ip_address(end_s.strip())
            for value in range(int(start), int(end) + 1):
                ip = str(ipaddress.ip_address(value))
                if ip not in used:
                    return ip
        else:
            network = ipaddress.ip_network(pool_range, strict=False)
            for ip_obj in network.hosts():
                ip = str(ip_obj)
                if ip not in used:
                    return ip
        raise SystemExit(f"No free IP found in MetalLB pool range {pool_range}")

    def prepare_server_state(self, *, run, state, config, ip_name, **kwargs):
        server_config = self._server_config(config)
        state["kubernetes_service_type"] = server_config.get("service_type", "ClusterIP")
        state["kubernetes_service_annotations"] = server_config.get("annotations") or {}
        state["kubernetes_load_balancer_ip"] = server_config.get("load_balancer_ip")
        public_service = self._public_service_config(config)
        if public_service:
            state["kubernetes_public_service"] = {
                "name": self._public_service_name(public_service),
                "namespace": self._public_service_namespace(config, public_service),
            }

    def release_ip(self, *, run, ip_name, state):
        print(f"No external IP to release for {self.name}.")

    def server_service_type(self, *, state):
        return state.get("kubernetes_service_type") or "ClusterIP"

    def server_service_helm_args(self, *, server_ip, state):
        args = []
        annotations = state.get("kubernetes_service_annotations") or {}
        if annotations:
            args += service_annotation_args(annotations)
        load_balancer_ip = state.get("kubernetes_load_balancer_ip")
        if load_balancer_ip:
            args += ["--set", f"service.loadBalancerIP={load_balancer_ip}"]
        return args

    def status_endpoint(self, *, state, config):
        server_config = self._server_config(config)
        public_service = self._public_service_config(config)
        if public_service:
            return "Server address", server_config.get("address") or public_service.get("load_balancer_ip") or "N/A"
        return "Server address", server_config.get("address") or self._default_server_address(config)

    def admin_endpoint(self, *, config, server_ip):
        server_config = self._server_config(config)
        admin_config = server_config.get("admin") or {}
        service_type = server_config.get("service_type", "ClusterIP")
        if admin_config.get("host") or admin_config.get("port"):
            return admin_config.get("host") or server_ip, int(admin_config.get("port") or 8003)
        if service_type == "ClusterIP":
            return "localhost", 18003
        return None

    def after_server_deploy(self, *, run, state, config, server):
        public_service = self._public_service_config(config)
        if not public_service:
            return
        server_ip = state.get("server_ip")
        if not server_ip:
            raise SystemExit("server_ip missing; cannot apply Kubernetes public service")

        namespace = self._public_service_namespace(config, public_service)
        name = self._public_service_name(public_service)
        port = int(public_service.get("port") or 8002)
        target_port = int(public_service.get("target_port") or port)
        annotations = public_service.get("annotations") or {}
        metallb = public_service.get("metallb") or {}
        pool = metallb.get("pool")
        if pool:
            annotations = {
                "metallb.universe.tf/address-pool": pool,
                "metallb.io/address-pool": pool,
                **annotations,
            }
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name, "namespace": namespace, "annotations": annotations},
            "spec": {
                "type": "LoadBalancer",
                "loadBalancerIP": server_ip,
                "selector": {"app.kubernetes.io/name": server.name},
                "ports": [
                    {
                        "name": public_service.get("port_name") or "fl-port",
                        "protocol": public_service.get("protocol") or "TCP",
                        "port": port,
                        "targetPort": target_port,
                    }
                ],
            },
        }
        print(f"Applying Kubernetes public service {namespace}/{name} ({server_ip}:{port}) ...")
        run(
            ["kubectl", "--kubeconfig", server.kubeconfig, "-n", namespace, "apply", "-f", "-"],
            input=json.dumps(manifest),
        )

    def cleanup_server_resources(self, *, run, state, config):
        public_service = self._public_service_config(config)
        if not public_service:
            return True
        server = self._server_participant(config)
        namespace = self._public_service_namespace(config, public_service)
        name = self._public_service_name(public_service)
        print(f"Deleting Kubernetes public service {namespace}/{name} ...")
        namespace_result = run(
            ["kubectl", "--kubeconfig", server.kubeconfig, "get", "ns", namespace],
            check=False,
            capture=True,
        )
        if namespace_result.returncode != 0:
            return True
        result = run(
            [
                "kubectl",
                "--kubeconfig",
                server.kubeconfig,
                "-n",
                namespace,
                "delete",
                "svc",
                name,
                "--ignore-not-found",
            ],
            check=False,
        )
        return result.returncode == 0
