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

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from cryptography import x509

from nvflare.lighter.constants import CtxKey, ParticipantType, PropKey
from nvflare.lighter.entity import Participant, Project, participant_from_dict
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import (
    Identity,
    generate_cert,
    generate_keys,
    serialize_cert,
    serialize_pri_key,
    verify_cert_chain,
)


@pytest.fixture
def project(tmp_path):
    example = Path(__file__).resolve().parents[3] / "examples/advanced/external_provisioning/project.yml"
    config = yaml.safe_load(example.read_text())
    next(b for b in config["builders"] if b["path"].endswith(".CertBuilder"))["args"]["public_root_cert"] = str(
        tmp_path / "rootCA.pem"
    )
    participants = [participant_from_dict(p) for p in config.pop("participants")]
    project = Project(config["name"], config["description"], participants=participants, props=config)
    listener = {"default_host": "localhost", "port": 9000, "conn_sec": "mtls"}
    project.get_clients()[0].set_prop("listening_host", listener)
    project.add_relay("relay", "org", {"external_cert": True, "listening_host": listener})
    key, pub = generate_keys()
    root = generate_cert(Identity("enterprise", "org"), Identity("enterprise", "org"), key, pub, ca=True)
    (tmp_path / "rootCA.pem").write_bytes(serialize_cert(root))
    return project, key, root


def provision(tmp_path, project):
    builders = (
        prepare_builders(project.props)
        if project.get_prop("builders")
        else [WorkspaceBuilder(), StaticFileBuilder(), CertBuilder(), SignatureBuilder()]
    )
    return Provisioner(str(tmp_path / "workspace"), builders).provision(project)


def test_public_root_distributed_without_issuance_or_reading_previous_keys(tmp_path, project):
    p, _, root = project
    with (
        patch("nvflare.lighter.impl.cert.generate_keys", side_effect=AssertionError("generated a key")),
        patch("nvflare.lighter.impl.cert._CertState", side_effect=AssertionError("read private-key state")),
        patch("nvflare.fuel.sec.admin_cert_provider.obtain_admin_cert_files", side_effect=AssertionError("enrollment")),
    ):
        for stage in range(2):
            ctx = provision(tmp_path, p)
            assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get(CtxKey.ERRORS)
            prod = Path(ctx[CtxKey.CURRENT_PROD_DIR])
            assert not list(prod.rglob("*.key"))
            assert not list(prod.rglob("*.crt"))
            assert not list(prod.rglob("signature.json"))
            assert ctx.get(CtxKey.ROOT_PRI_KEY) is None
            for participant in p.get_all_participants():
                assert (prod / participant.name / "startup/rootCA.pem").read_bytes() == serialize_cert(root)
            relay_readme = (prod / "relay/readme.txt").read_text()
            assert all(name in relay_readme for name in ("client.crt", "client.key", "server.crt", "server.key"))
            admin = json.loads((prod / "admin-sso/startup/fed_admin.json").read_text())
            assert "admin_cert_provider" in admin["admin"]
            file_admin = json.loads((prod / "automation@example.com/startup/fed_admin.json").read_text())["admin"]
            assert file_admin["client_cert"] == "client.crt"
            assert file_admin["client_key"] == "client.key"
            assert file_admin["username"] == "automation@example.com"
            assert "admin_cert_provider" not in file_admin
            if stage == 0:
                # Deployment-installed credentials and old legacy state are never imported.
                (prod / "server/startup/server.key").write_text("deployment-owned key")
                (tmp_path / "workspace/demo/state/cert.json").write_text("invalid legacy state")


@pytest.mark.parametrize(
    "participant,setting",
    [
        ("server", "external_cert"),
        ("server", "external_job_ca"),
        ("site-1", "external_cert"),
        ("relay", "external_cert"),
        ("admin-sso", "admin_cert_provider"),
    ],
)
def test_required_external_sources_fail_before_issuance(tmp_path, project, participant, setting):
    p, _, _ = project
    next(x for x in p.get_all_participants() if x.name == participant).set_prop(setting, None)
    with patch("nvflare.lighter.impl.cert.generate_keys", side_effect=AssertionError("generated a key")):
        ctx = provision(tmp_path, p)
    assert ctx[CtxKey.BUILD_ERROR]
    assert f"{participant}: public-root-only provisioning requires {setting}" in str(ctx[CtxKey.ERRORS])
    assert not list((tmp_path / "workspace/demo").glob("prod_*"))


@pytest.mark.parametrize("value", ["", "   ", False, {}, ["rootCA.pem"]])
def test_invalid_public_root_argument_rejected(value):
    with pytest.raises(ValueError, match="public_root_cert must be a non-empty certificate file path"):
        CertBuilder(public_root_cert=value)


@pytest.mark.parametrize(
    "invalid,error",
    [
        ("private_key", "must contain only PEM certificate data"),
        ("multiple", "must contain exactly one public trust root"),
        ("expired", "public trust root is not currently valid"),
        ("signature", "must have a valid self-signature"),
        ("malformed", "public_root_cert is not a valid PEM certificate"),
    ],
)
def test_invalid_public_roots_are_rejected(tmp_path, project, invalid, error):
    p, key, root = project
    pem = serialize_cert(root)
    if invalid == "private_key":
        pem += serialize_pri_key(key)
    elif invalid == "multiple":
        pem *= 2
    elif invalid == "malformed":
        pem = b"-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----\n"
    else:
        kwargs = {}
        if invalid == "expired":
            now = datetime.datetime.now(datetime.timezone.utc)
            kwargs = {
                "not_valid_before": now - datetime.timedelta(days=2),
                "not_valid_after": now - datetime.timedelta(days=1),
            }
        signing_key = generate_keys()[0] if invalid == "signature" else key
        pem = serialize_cert(
            generate_cert(
                Identity("enterprise", "org"),
                root.subject,
                signing_key,
                key.public_key(),
                ca=True,
                **kwargs,
            )
        )
    (tmp_path / "rootCA.pem").write_bytes(pem)
    ctx = provision(tmp_path, p)
    assert ctx[CtxKey.BUILD_ERROR]
    assert error in str(ctx[CtxKey.ERRORS])
    assert not list((tmp_path / "workspace/demo").glob("prod_*"))


def test_public_mode_rejects_private_root_and_content_signing(tmp_path, project):
    p, key, _ = project
    p.root_private_key = key
    ctx = provision(tmp_path, p)
    assert ctx[CtxKey.BUILD_ERROR] and "legacy project root/key" in str(ctx[CtxKey.ERRORS])
    p.root_private_key = None
    p.get_server().set_prop(PropKey.CC_ENABLED, True)
    ctx = provision(tmp_path, p)
    assert ctx[CtxKey.BUILD_ERROR] and "cannot sign CC content" in str(ctx[CtxKey.ERRORS])


@pytest.mark.parametrize("supplied_root", [False, True])
def test_legacy_and_supplied_root_plus_key_still_issue_credentials(tmp_path, project, supplied_root):
    _, key, _ = project
    # Legacy issuance uses a CN-only root; enterprise subjects use public-root mode.
    root = generate_cert(Identity("legacy"), Identity("legacy"), key, key.public_key(), ca=True)
    p = Project(
        "legacy",
        "legacy modes",
        participants=[
            Participant(ParticipantType.SERVER, "server", "org"),
            Participant(ParticipantType.CLIENT, "site-1", "org"),
            Participant(ParticipantType.ADMIN, "admin@example.com", "org", {"role": "project_admin"}),
        ],
        serialized_root_cert=serialize_cert(root) if supplied_root else None,
        root_private_key=key if supplied_root else None,
    )
    ctx = provision(tmp_path, p)
    assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get(CtxKey.ERRORS)
    prod = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    if supplied_root:
        assert (prod / "server/startup/rootCA.pem").read_bytes() == serialize_cert(root)
    for name, basename in (("server", "server"), ("site-1", "client"), ("admin@example.com", "client")):
        kit = prod / name / "startup"
        cert = x509.load_pem_x509_certificate((kit / f"{basename}.crt").read_bytes())
        verify_cert_chain(cert, [], ctx[CtxKey.ROOT_CERT])
        assert (kit / f"{basename}.key").is_file()
    assert (prod / "server/startup/job_ca.key").is_file()


@pytest.mark.parametrize("invalid", ["missing_bc", "missing_ku", "noncritical_bc", "not_ca", "no_key_cert_sign"])
def test_invalid_root_extensions_have_actionable_errors(tmp_path, project, invalid):
    from cryptography.hazmat.primitives import hashes

    p, key, root = project
    builder = (
        x509.CertificateBuilder()
        .subject_name(root.subject)
        .issuer_name(root.issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(root.not_valid_before_utc)
        .not_valid_after(root.not_valid_after_utc)
    )
    for extension in root.extensions:
        value, critical = extension.value, extension.critical
        if isinstance(value, x509.BasicConstraints):
            if invalid == "missing_bc":
                continue
            if invalid == "noncritical_bc":
                critical = False
            if invalid == "not_ca":
                value = x509.BasicConstraints(ca=False, path_length=None)
        elif isinstance(value, x509.KeyUsage):
            if invalid == "missing_ku":
                continue
            if invalid == "no_key_cert_sign":
                value = x509.KeyUsage(True, False, False, False, False, False, True, False, False)
        builder = builder.add_extension(value, critical)
    (tmp_path / "rootCA.pem").write_bytes(serialize_cert(builder.sign(key, hashes.SHA256())))
    ctx = provision(tmp_path, p)
    assert ctx[CtxKey.BUILD_ERROR]
    error = (
        "requires CA BasicConstraints and KeyUsage with keyCertSign"
        if invalid.startswith("missing_")
        else "public trust root must be a CA with keyCertSign"
    )
    assert error in str(ctx[CtxKey.ERRORS])
    assert not list((tmp_path / "workspace/demo").glob("prod_*"))


@pytest.mark.parametrize("path_length", [0, 1, None])
def test_public_root_requires_room_for_job_ca(tmp_path, project, path_length):
    p, key, _ = project
    root = generate_cert(
        Identity("enterprise", "org"),
        Identity("enterprise", "org"),
        key,
        key.public_key(),
        ca=True,
        ca_path_length=path_length,
    )
    (tmp_path / "rootCA.pem").write_bytes(serialize_cert(root))
    ctx = provision(tmp_path, p)
    if path_length == 0:
        assert ctx[CtxKey.BUILD_ERROR]
        assert "path_length=0 cannot support the required subordinate job CA" in str(ctx[CtxKey.ERRORS])
        assert not list((tmp_path / "workspace/demo").glob("prod_*"))
    else:
        assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get(CtxKey.ERRORS)
        kit = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / "server/startup"
        assert (kit / "rootCA.pem").read_bytes() == serialize_cert(root)


@pytest.mark.parametrize("public_first", [True, False])
def test_public_root_rejects_duplicate_cert_builders(tmp_path, project, public_first):
    public = CertBuilder(public_root_cert=str(tmp_path / "rootCA.pem"))
    builders = [public, CertBuilder()] if public_first else [CertBuilder(), public]
    with patch("nvflare.lighter.impl.cert.generate_keys", side_effect=AssertionError("legacy issuance")):
        ctx = Provisioner(str(tmp_path / "workspace"), [WorkspaceBuilder(), *builders]).provision(project[0])
    assert ctx[CtxKey.BUILD_ERROR]
    assert "requires exactly one CertBuilder" in str(ctx[CtxKey.ERRORS])


@pytest.mark.parametrize("before_cert", [True, False])
@pytest.mark.parametrize("kind", ["CC", "HE"])
def test_signing_builders_reject_public_root_independent_of_order(tmp_path, project, before_cert, kind):
    import importlib.util
    from unittest.mock import MagicMock

    if kind == "CC":
        from nvflare.lighter.cc_provision.impl.cc import CCBuilder

        builder_type = CCBuilder
    else:
        # Exercise the actual HE guard without requiring native TenSEAL on every Python version.
        spec = importlib.util.spec_from_file_location(
            "he_guard_test", Path(__file__).parents[3] / "nvflare/lighter/impl/he.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"tenseal": MagicMock()}):
            spec.loader.exec_module(module)
        builder_type = module.HEBuilder

    class CustomSigningBuilder(builder_type):
        pass

    signing = CustomSigningBuilder()

    class CustomCertBuilder(CertBuilder):
        pass

    cert = CustomCertBuilder(public_root_cert=str(tmp_path / "rootCA.pem"))
    builders = [signing, cert] if before_cert else [cert, signing]
    with patch.object(signing, "build", side_effect=AssertionError("signing builder ran")):
        ctx = Provisioner(str(tmp_path / "workspace"), [WorkspaceBuilder(), *builders]).provision(project[0])
    assert ctx[CtxKey.BUILD_ERROR]
    assert f"does not support {kind} content signing" in str(ctx[CtxKey.ERRORS])


def test_public_root_edge_rejected_before_building(project):
    from nvflare.lighter.provision import provision_for_edge

    with pytest.raises(ValueError, match="not supported by hierarchical edge"):
        provision_for_edge(None, project[0].props)


def test_cli_resolves_root_relative_to_project_file(tmp_path, project, monkeypatch):
    from types import SimpleNamespace

    from nvflare.lighter.provision import handle_provision

    config = yaml.safe_load(
        (Path(__file__).parents[3] / "examples/advanced/external_provisioning/project.yml").read_text()
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(yaml.safe_dump(config))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    args = SimpleNamespace(
        project_file=str(project_file),
        workspace=str(tmp_path / "cli-workspace"),
        generate=False,
        gen_edge=False,
        force=True,
        custom_folder=".",
        add_user="",
        add_client="",
        gen_scripts=False,
    )
    with patch("nvflare.tool.install_skills.install_skills"):
        handle_provision(args)
    kits = list((tmp_path / "cli-workspace/demo").glob("prod_*/server/startup/rootCA.pem"))
    assert len(kits) == 1
    assert kits[0].read_bytes() == serialize_cert(project[2])


def test_python_api_resolves_public_root_from_current_directory(tmp_path, project, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = Provisioner(
        str(tmp_path / "api-workspace"),
        [WorkspaceBuilder(), StaticFileBuilder(), CertBuilder(public_root_cert="rootCA.pem"), SignatureBuilder()],
    ).provision(project[0])
    assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get(CtxKey.ERRORS)
    kit = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / "server/startup"
    assert (kit / "rootCA.pem").read_bytes() == serialize_cert(project[2])
    assert not list(kit.glob("*.key"))


@pytest.mark.parametrize("public_root", [False, True])
def test_admin_credential_sources_are_mutually_exclusive(tmp_path, project, public_root):
    p, _, _ = project
    if not public_root:
        p.set_prop("builders", None)
    p.get_admins()[0].set_prop(PropKey.EXTERNAL_CERT, True)
    ctx = provision(tmp_path, p)
    assert ctx[CtxKey.BUILD_ERROR]
    assert "external_cert and admin_cert_provider are mutually exclusive" in str(ctx[CtxKey.ERRORS])
