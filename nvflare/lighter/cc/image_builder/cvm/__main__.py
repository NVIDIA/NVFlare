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

"""Build, distribute and administer CVMs and application vaults."""

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from .artifacts import oci
from .artifacts.acceptance import aggregate as aggregate_acceptance
from .artifacts.bundle import approve_bundle
from .artifacts.packaging import package_bundle
from .build import config, cvm, vault
from .common.contracts import PLATFORMS
from .common.errors import BuildError, ConfigurationError
from .common.io import read_json, write_json
from .common.linux import enable_diagnostics
from .host.preflight import check_host
from .trustee import admin
from .trustee.client import delete_resource, resource_role_acl
from .trustee.import_references import import_references
from .trustee.inspect_tcb import inspect_tcb
from .trustee.preflight import check_trustee
from .trustee.provenance import provenance


def report_bundle(result):
    print(f"Generic bundle: {result}")
    manifest_name = "cvm_manifest.json" if (result / "cvm_manifest.json").is_file() else "cvm_manifest.pending.json"
    manifest = read_json(result / manifest_name)
    artifact = result.parent / f"cvm_{manifest['profile_version']}_{manifest['platform']}.oci.tar"
    print(f"OCI artifact: {artifact}")
    if (result / "approval.json").is_file():
        print("The exact finalized bundle passed the acceptance runner and is approved.")
    elif manifest_name == "cvm_manifest.json" and not manifest.get("dev_mode"):
        print("Production approval remains pending; use cvmctl admin approve after acceptance testing.")


def build(args):
    report_bundle(
        cvm.build(
            args.config,
            args.platform,
            args.output,
            defer_measurements=args.defer_measurements,
            gpu=args.gpu,
            dev=args.dev,
            acceptance_runner=args.acceptance_runner,
            approval_key=args.approval_key,
        )
    )


def build_vault(args):
    result = vault.build(args.config, args.output, args.candidate, args.dev, args.plain_http, args.project_config)
    print(f"Vault staging: {result}")
    for name in sorted(read_json(result / "oci_artifacts.json")["artifacts"]):
        print(f"OCI artifact: {result / name}")


def pull(args):
    local = Path(args.source).expanduser().is_file()
    if local:
        if args.archive_sha256 is None and not args.allow_unverified:
            raise BuildError(
                "Local OCI archives require --archive-sha256 from the publisher's authenticated channel, "
                "or an explicit --allow-unverified"
            )
    elif args.cosign_key is None and not args.allow_unverified:
        raise BuildError("Registry artifacts require --cosign-key for signature verification, or --allow-unverified")
    output, descriptor, config = oci.materialize(
        args.source,
        args.output,
        args.merge,
        args.plain_http,
        archive_sha256=args.archive_sha256,
        cosign_key=args.cosign_key,
    )
    print(f"Materialized {descriptor['artifactType']} {descriptor['digest']} at {output}")
    if args.allow_unverified and (args.archive_sha256 is None and args.cosign_key is None):
        print("Publisher authentication was skipped; the digest alone does not identify who selected it.")
    if config.get("launch_directory"):
        print(f"Launch: cd {output / config['launch_directory']} && sudo ./launch_cvm.sh")


def approve(args):
    approve_bundle(args.bundle, read_json(args.evidence), args.signing_key)
    package_bundle(args.bundle)


def acceptance_report(args):
    report = aggregate_acceptance(args.bundle, args.evidence, args.evidence_key)
    write_json(args.output, report, mode=0o600)
    print(f"Acceptance report: {Path(args.output).resolve()}")


def host_preflight(args):
    if not check_host(args.firmware, args.quote_probe):
        print(
            "Quote generation NOT checked. Run --quote-probe with a site smoke test before building. "
            "Check collateral access and multi-package platform registration if quotes are empty.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--diagnostics",
        help="private directory that retains stderr of failed commands which never handle secrets",
    )
    commands = root.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("build", help="construct a generic CVM")
    stage1.add_argument("config", nargs="?", default=str(config.SOURCE / "config/cvm_profile.yml"))
    stage1.add_argument("-p", "--platform", choices=PLATFORMS)
    stage1.add_argument("--output")
    stage1.add_argument("--defer-measurements", action="store_true", help="collect references on the target host later")
    stage1.add_argument("--dev", action="store_true", help="separate dev- profile without TEE or KBS authorization")
    stage1.add_argument("--gpu", action="append", help="repeat for each explicit NVIDIA GPU PCI address")
    stage1.add_argument("--acceptance-runner", help="trusted executable called as RUNNER BUNDLE REPORT")
    stage1.add_argument(
        "--approval-key",
        help="Ed25519 private key that signs approval.json after site acceptance (or profile approval_signing_key)",
    )
    stage1.set_defaults(handler=build)

    finalize = commands.add_parser("finalize", help="measure a previously constructed bundle")
    finalize.add_argument("bundle")
    finalize.add_argument("--reference-evidence", help="private reference report captured on a trusted target host")
    finalize.add_argument("--gpu", action="append", help="repeat for each explicit NVIDIA GPU PCI address")
    finalize.set_defaults(handler=lambda a: report_bundle(cvm.finalize(a.bundle, a.reference_evidence, a.gpu)))

    stage2 = commands.add_parser("vault", help="seal an application vault")
    stage2.add_argument("config")
    stage2.add_argument("--output")
    stage2.add_argument("--project-config", help="default: nearest cvm_project.yml above the build YAML directory")
    stage2.add_argument("--plain-http", action="store_true", help="allow an unencrypted test registry connection")
    stage2.add_argument("--candidate", action="store_true", help="acceptance-only vault for an exact unapproved bundle")
    stage2.add_argument("--dev", action="store_true", help="plain ext4 for a separate dev- profile; no KBS")
    stage2.set_defaults(handler=build_vault)

    acceptance = commands.add_parser(
        "acceptance-report", help="aggregate exact-manifest site evidence for admin approve"
    )
    acceptance.add_argument("bundle")
    acceptance.add_argument("evidence", nargs="+", help="result.json files or directories containing them")
    acceptance.add_argument(
        "--evidence-key",
        action="append",
        required=True,
        help="trusted Ed25519 public key for acceptance result producers; repeat to rotate keys",
    )
    acceptance.add_argument("--output", required=True)
    acceptance.set_defaults(handler=acceptance_report)

    materialize = commands.add_parser("pull", help="materialize a local OCI tar or immutable registry artifact")
    materialize.add_argument("source")
    materialize.add_argument("--output")
    materialize.add_argument("--merge", action="store_true", help="add a finalized platform to an existing profile")
    materialize.add_argument("--plain-http", action="store_true", help="allow an unencrypted test registry connection")
    materialize.add_argument("--archive-sha256", help="published SHA-256 of a local .oci.tar (authenticated channel)")
    materialize.add_argument("--cosign-key", help="cosign public key that must have signed the registry digest")
    materialize.add_argument(
        "--allow-unverified",
        action="store_true",
        help="skip publisher authentication (lab use only; digests alone do not identify the publisher)",
    )
    materialize.set_defaults(handler=pull)
    publish = commands.add_parser("publish", help="copy a local OCI tar to a registry")
    publish.add_argument("source")
    publish.add_argument("destination")
    publish.add_argument("--plain-http", action="store_true", help="allow an unencrypted test registry connection")
    publish.add_argument("--cosign-key", help="cosign private key used to sign the published immutable digest")
    publish.set_defaults(
        handler=lambda a: print(
            f"Published: {oci.publish(a.source, a.destination, a.plain_http, cosign_key=a.cosign_key)}"
        )
    )

    administration = commands.add_parser("admin", help="approve, install, retire or revoke")
    actions = administration.add_subparsers(required=True)
    approval = actions.add_parser("approve")
    approval.add_argument("bundle")
    approval.add_argument("evidence")
    approval.add_argument("--signing-key", required=True, help="Ed25519 private key of the acceptance authority")
    approval.set_defaults(handler=approve)
    install = actions.add_parser("install")
    install.add_argument("config", help="administration JSON; production requires approval_public_keys")
    install.add_argument("bundle")
    install.add_argument("--candidate", action="store_true", help="install an exact unapproved bundle for acceptance")
    install.set_defaults(handler=lambda a: admin.install(read_json(a.config), a.bundle, a.candidate))
    retire = actions.add_parser("retire")
    retire.add_argument("config")
    retire.add_argument("build_id")
    retire.set_defaults(handler=lambda a: admin.retire(read_json(a.config), a.build_id))
    revoke = actions.add_parser("revoke")
    revoke.add_argument(
        "config", help="use a cvm-resources admin_token_file, or admin_private_key with admin_role=cvm-resources"
    )
    revoke.add_argument("resource")
    revoke.set_defaults(handler=lambda a: delete_resource(read_json(a.config), a.resource))
    acl = actions.add_parser("acl", help="print the KBS ACL entry confining a resource role to one bundle")
    acl.add_argument("build_id")
    acl.set_defaults(handler=lambda a: print(json.dumps(resource_role_acl(a.build_id), indent=2)))

    references = commands.add_parser("references", help="import reviewed references into a stopped Trustee")
    references.add_argument("bundle", type=Path)
    references.add_argument("--store", type=Path, required=True, help="Trustee local_fs reference_value directory")
    references.add_argument("--state", type=Path, required=True)
    references.add_argument("--expires", required=True, help="approved UTC expiry, e.g. 2026-12-01T00:00:00Z")
    references.set_defaults(handler=lambda a: import_references(a.bundle, a.store, a.state, a.expires))
    record = commands.add_parser("provenance", help="record a clean upstream Trustee revision and a built binary")
    record.add_argument("source", type=Path)
    record.add_argument("binary", type=Path, help="kbs or kbs-client executable built from that checkout")
    record.add_argument("output", type=Path)
    record.set_defaults(handler=lambda a: write_json(a.output, provenance(a.source, a.binary)))
    inspect = commands.add_parser("inspect-tcb", help="inspect candidate TDX TCB fields without approving them")
    inspect.add_argument("evidence", type=Path)
    inspect.set_defaults(handler=lambda a: print(json.dumps(inspect_tcb(read_json(a.evidence)), indent=2)))

    preflight = commands.add_parser("preflight", help="check host or Trustee prerequisites")
    checks = preflight.add_subparsers(required=True)
    host = checks.add_parser("host")
    host.add_argument("--firmware", type=Path, required=True)
    host.add_argument("--quote-probe", type=Path, help="executable that boots a TD and verifies a nonempty quote")
    host.set_defaults(handler=host_preflight)
    trustee = checks.add_parser("trustee")
    trustee.set_defaults(handler=lambda a: check_trustee())
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.diagnostics:
            enable_diagnostics(args.diagnostics)
        args.handler(args)
    except ConfigurationError as exc:
        cli.exit(1, f"Invalid configuration: {exc}\n")
    except (BuildError, OSError, ValueError, KeyError, tarfile.TarError, subprocess.SubprocessError) as exc:
        if args.command == "vault":
            # Application inputs and key-upload failures can contain secrets.
            cli.exit(
                1,
                "Vault build failed; retain any build_failure.json and resolve uncertain key uploads before retrying\n",
            )
        cli.exit(1, f"CVM operation failed: {exc}\n")


if __name__ == "__main__":
    main()
