from __future__ import annotations

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

    def validate_server_config(self, config):
        server_config = self._server_config(config)
        service_type = server_config.get("service_type", "ClusterIP")
        if service_type not in {"ClusterIP", "LoadBalancer", "NodePort"}:
            raise SystemExit(f"clouds.{self.name}.server.service_type must be ClusterIP, LoadBalancer, or NodePort")

    def reserve_ip(self, *, run, ip_tag, config=None, **kwargs):
        server_config = self._server_config(config)
        address = server_config.get("address") or self._default_server_address(config)
        print(f"Using Kubernetes service address {address} ({self.name})")
        return address, ip_tag

    def prepare_server_state(self, *, run, state, config, ip_name, **kwargs):
        server_config = self._server_config(config)
        state["kubernetes_service_type"] = server_config.get("service_type", "ClusterIP")
        state["kubernetes_service_annotations"] = server_config.get("annotations") or {}
        state["kubernetes_load_balancer_ip"] = server_config.get("load_balancer_ip")

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
