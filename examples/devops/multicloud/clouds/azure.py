import yaml

from .base import CloudProvider, service_annotation_args


class AzureProvider(CloudProvider):
    name = "azure"
    auth_check_cmd = ["az", "account", "show"]
    auth_failed_message = "Azure auth failed. Run: az login"
    auth_expired_message = "Azure session expired. Run: az login"

    def parse_kubeconfig(self, kc_path):
        data = yaml.safe_load(kc_path.read_text())
        current_ctx = data.get("current-context")
        if not current_ctx:
            raise ValueError(f"{kc_path}: no current-context")
        if not any(c.get("name") == current_ctx for c in data.get("contexts", [])):
            raise ValueError(f"{kc_path}: current-context {current_ctx!r} not found in contexts")
        return {}

    def validate_server_config(self, config):
        if not config.azure_resource_group:
            raise SystemExit("clouds.azure.resource_group is required when the server is in Azure")
        if not config.azure_location:
            raise SystemExit("clouds.azure.location is required when the server is in Azure")

    def reserve_ip(self, *, run, ip_tag, azure_resource_group=None, azure_location=None, **kwargs):
        if not isinstance(azure_resource_group, str) or not azure_resource_group.strip():
            raise ValueError(
                "Azure static IP reservation requires a non-empty resource_group. "
                "Set clouds.azure.resource_group in the deploy config."
            )
        if not isinstance(azure_location, str) or not azure_location.strip():
            raise ValueError(
                "Azure static IP reservation requires a non-empty location. "
                "Set clouds.azure.location in the deploy config."
            )
        print(f"Reserving Azure Public IP {ip_tag} ...")
        run(
            [
                "az",
                "network",
                "public-ip",
                "create",
                "--resource-group",
                azure_resource_group,
                "--name",
                ip_tag,
                "--sku",
                "Standard",
                "--allocation-method",
                "Static",
                "--location",
                azure_location,
            ],
            check=False,
        )
        r = run(
            [
                "az",
                "network",
                "public-ip",
                "show",
                "--resource-group",
                azure_resource_group,
                "--name",
                ip_tag,
                "--query",
                "ipAddress",
                "--output",
                "tsv",
            ],
            capture=True,
        )
        ip = r.stdout.strip()
        if not ip:
            raise RuntimeError(f"az network public-ip show returned no IP for {ip_tag}")
        print(f"  Reserved: {ip} ({ip_tag})")
        return ip, ip_tag

    def release_ip(self, *, run, ip_name, state):
        resource_group = state.get("azure_resource_group")
        if not isinstance(resource_group, str) or not resource_group.strip():
            raise ValueError(
                "Azure static IP release requires a non-empty resource_group. "
                "Pass a config that includes clouds.azure.resource_group."
            )
        print(f"Releasing Azure Public IP {ip_name} ...")
        r = run(
            ["az", "network", "public-ip", "delete", "--resource-group", resource_group, "--name", ip_name],
            check=False,
        )
        if r.returncode != 0:
            detail = ""
            stderr = getattr(r, "stderr", "") or ""
            if stderr.strip():
                detail = f": {stderr.strip()}"
            print(
                f"  Warning: failed to delete Azure Public IP {ip_name} in resource group {resource_group}{detail}. "
                "The IP may still be allocated and require manual cleanup."
            )

    def server_service_helm_args(self, *, server_ip, state):
        azure_pip_name = state.get("azure_pip_name")
        azure_resource_group = state.get("azure_resource_group")
        if not azure_pip_name or not azure_resource_group:
            raise RuntimeError("Azure server requires azure_pip_name and azure_resource_group")
        return service_annotation_args(
            {
                "service.beta.kubernetes.io/azure-pip-name": azure_pip_name,
                "service.beta.kubernetes.io/azure-load-balancer-resource-group": azure_resource_group,
            }
        )
