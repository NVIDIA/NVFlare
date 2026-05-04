class CloudProvider:
    name = ""
    auth_check_cmd: list[str] = []
    auth_failed_message = ""
    auth_expired_message = ""

    def parse_kubeconfig(self, kc_path):
        return {}

    def validate_server_config(self, config):
        pass

    def reserve_ip(self, *, run, ip_tag, **kwargs):
        raise NotImplementedError

    def prepare_server_state(self, *, run, state, config, ip_name, **kwargs):
        pass

    def release_ip(self, *, run, ip_name, state):
        raise NotImplementedError

    def server_service_helm_args(self, *, server_ip, state):
        raise NotImplementedError


def service_annotation_args(annotations: dict[str, str]) -> list[str]:
    args = []
    for k, v in annotations.items():
        escaped = k.replace(".", r"\.")
        args += ["--set-string", f"service.annotations.{escaped}={v}"]
    return args
