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

"""nvflare catalog: read the datasets of a study's sites into the federated catalog, and show the catalog."""

import argparse
import os
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    AuthorizationError,
    CommandError,
    InternalError,
    InvalidArgumentError,
    InvalidJobDefinition,
    JobNotFound,
    MonitorReturnCode,
    NoConnection,
    SessionClosed,
)
from nvflare.fuel.flare_api.job_status import job_status_outcome
from nvflare.tool.cli_output import (
    is_json_mode,
    output_error,
    output_error_message,
    output_ok,
    output_usage_error,
    print_human,
)
from nvflare.tool.cli_session import add_startup_kit_selection_args

CMD_CATALOG_SHOW = "show"
CMD_CATALOG_READ = "read"
_NO_RETRY_TOKEN_SCHEMA = {"supported": False}

_catalog_sub_cmd_parsers = {}
_catalog_handlers = {}
_catalog_root_parser = None


# ---- session and errors ------------------------------------------------------------------------------------


@contextmanager
def _catalog_session(args):
    from nvflare.tool.cli_output import get_validated_connect_timeout
    from nvflare.tool.cli_session import new_cli_session_for_args

    try:
        sess = new_cli_session_for_args(args=args, timeout=get_validated_connect_timeout(), study=args.study)
    except ValueError as e:
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=str(e), hint=getattr(e, "hint", None))
        raise SystemExit(4)
    try:
        yield sess
    finally:
        sess.close()


def _handle_command_error(e: Exception):
    if isinstance(e, CommandError):
        output_error_message(e.error_code, e.message, hint=e.hint, exit_code=e.exit_code)
    elif isinstance(e, AuthorizationError):
        output_error_message(
            "NOT_AUTHORIZED", str(e) or "not authorized", hint="Use a startup kit with the required role.", exit_code=1
        )
    elif isinstance(e, AuthenticationError):
        output_error("AUTH_FAILED", exit_code=2, detail=str(e))
    elif isinstance(e, (NoConnection, SessionClosed)):
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
    elif isinstance(e, JobNotFound):
        output_error_message("JOB_NOT_FOUND", str(e), hint="Run 'nvflare job list' to find the job.", exit_code=1)
    elif isinstance(e, TimeoutError):
        output_error("TIMEOUT", exit_code=3, detail=str(e))
    else:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))


def _run(compute, parser, render):
    """Output what compute() returns, or its error as the shared CLI error output."""
    try:
        data = compute()
    except (ValueError, InvalidArgumentError, InvalidJobDefinition) as e:
        output_usage_error(parser, detail=str(e), exit_code=4)
        return
    except Exception as e:
        _handle_command_error(e)
        return
    render(data)


# ---- show --------------------------------------------------------------------------------------------------


def _summary(entry) -> dict:
    row = {
        "site": entry.client,
        "dataset": entry.dataset,
        "available": entry.available,
        "last_checked_at": entry.last_checked_at,
    }
    if entry.catalog is not None:
        row.update(
            display_name=entry.catalog.display_name,
            catalog_version=entry.catalog_version,
            schema_version=entry.schema_version,
            published_at=entry.published_at,
        )
    return row


def _site_health(h) -> dict:
    return {
        "site": h.client,
        "health": h.health,
        "reason_code": h.reason_code,
        "last_success_at": h.last_success_at,
        "stale": h.stale,
    }


def _detail(entry) -> dict:
    detail = _summary(entry)
    if entry.catalog is not None:
        detail["metadata"] = entry.catalog.metadata_wire()
        detail["observed_at"] = entry.catalog.observed_at
    return detail


def _show(api, args) -> dict:
    filters = {k: v for k, v in {"client": args.site, "dataset": args.dataset}.items() if v is not None}
    entries = list(api.search(**filters))
    sites = [_site_health(h) for h in api.status()]
    if args.overview:
        return {"study": args.study, "total": len(entries), "datasets": [_summary(e) for e in entries], "sites": sites}
    if args.site is not None and args.dataset is not None:
        if not entries:
            raise CommandError(
                error_code="DATASET_NOT_FOUND",
                message=f"no dataset '{args.dataset}' at site '{args.site}'",
                hint="Run 'nvflare catalog show --overview' to list the datasets of the study.",
            )
        return {"study": args.study, "dataset": _detail(entries[0]), "sites": sites}
    return {"study": args.study, "total": len(entries), "datasets": [_detail(e) for e in entries], "sites": sites}


def _table(headers, rows):
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    print_human("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print_human("  ".join("-" * w for w in widths))
    for r in rows:
        print_human("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def _warn_about_sites(sites):
    for s in sites:
        if s["health"] in ("FAILED", "DEGRADED") or s["stale"]:
            if s["health"] == "FAILED":
                why = f"last read failed ({s['reason_code']})"
            elif s["health"] == "DEGRADED":
                why = f"last read was incomplete ({s['reason_code']}); some datasets may be hidden"
            else:
                why = "data may be out of date"
            hint = {
                "BUDGET_EXCEEDED": "Select fewer datasets or ask the site operator to adjust catalog read limits.",
                "POLICY_UNAVAILABLE": "Ask the site operator to check the catalog configuration and Study policy.",
                "INTERNAL": "Ask the site operator to check the catalog connector logs.",
            }.get(s["reason_code"], "Run a catalog read to refresh the site.")
            print_human(f"warning: site {s['site']}: {why}; last success {s['last_success_at'] or 'never'}. {hint}")


def _render_dataset(detail: dict):
    print_human(f"{detail['site']} / {detail['dataset']}  ({'available' if detail['available'] else 'unavailable'})")
    print_human(f"last checked: {detail['last_checked_at'] or '-'}")
    metadata = detail.get("metadata")
    if metadata is None:
        print_human("metadata is not available while the dataset is unavailable")
        return
    print_human(f"display name: {metadata['display_name']}")
    print_human(
        f"version: {detail['catalog_version']} (schema {detail['schema_version']}), published {detail['published_at']}"
    )
    for key in ("description", "owner", "source_version"):
        if metadata.get(key):
            print_human(f"{key.replace('_', ' ')}: {metadata[key]}")
    profile = {k: v for k, v in metadata["semantic_profile"].items() if v}
    if profile:
        print_human("semantic profile: " + ", ".join(f"{k}={v}" for k, v in profile.items()))
    for table in metadata["tables"]:
        print_human("")
        pk = f"  primary key: {', '.join(table['primary_key'])}" if table["primary_key"] else ""
        print_human(f"table {table['name']}{pk}")
        _table(
            ("COLUMN", "TYPE", "NULLABLE"),
            [(c["name"], c["type"], {True: "yes", False: "no", None: "-"}[c["nullable"]]) for c in table["columns"]],
        )
        for r in table["relationships"]:
            print_human(f"  {', '.join(r['columns'])} -> {r['target_table']}({', '.join(r['target_columns'])})")
    for item in metadata["lineage"]:
        print_human(f"derived from: {item['dataset']} ({item['source_version'] or 'unversioned'})")
    print_human(f"source: {metadata['provenance']['source']}")


def _render_show(data: dict, overview=False):
    if is_json_mode():
        _warn_about_sites(data["sites"])
        output_ok(data)
        return
    if "dataset" in data:
        _render_dataset(data["dataset"])
    else:
        if not data["datasets"]:
            print_human("no datasets")
        elif overview:
            _table(
                ("SITE", "DATASET", "NAME", "AVAILABLE", "VERSION", "LAST CHECKED"),
                [
                    (
                        d["site"],
                        d["dataset"],
                        d.get("display_name", "-"),
                        "yes" if d["available"] else "no",
                        str(d.get("catalog_version", "-")),
                        d["last_checked_at"] or "-",
                    )
                    for d in data["datasets"]
                ],
            )
        else:
            for i, detail in enumerate(data["datasets"]):
                if i:
                    print_human("")
                _render_dataset(detail)
    _warn_about_sites(data["sites"])


def cmd_show(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_catalog_parsers()
    handle_schema_flag(
        _catalog_sub_cmd_parsers[CMD_CATALOG_SHOW],
        "nvflare catalog show",
        [
            "nvflare catalog show --format json",
            "nvflare catalog show --overview",
            "nvflare catalog show --site site-a --dataset omop",
        ],
        sys.argv[1:],
        output_modes=["json"],
        streaming=False,
        mutating=False,
        idempotent=True,
        retry_token=_NO_RETRY_TOKEN_SCHEMA,
    )
    from nvflare.app_common.fdn.catalog.admin.catalog_api import CatalogAdminApi

    def compute():
        with _catalog_session(args) as sess:
            return _show(CatalogAdminApi(sess), args)

    _run(compute, _catalog_sub_cmd_parsers[CMD_CATALOG_SHOW], lambda data: _render_show(data, overview=args.overview))


# ---- read --------------------------------------------------------------------------------------------------


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = list(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))
    if not items:
        raise ValueError("--sites must name at least one site; omit it to read every site of the Study")
    return items


def _dataset_map(values: list[str] | None) -> dict[str, list[str]] | None:
    """Positional SITE:NAME arguments as {site: [names]}."""
    result: dict[str, list[str]] = {}
    for value in values or []:
        site, sep, name = value.partition(":")
        if not (sep and site.strip() and name.strip()):
            raise ValueError(f"dataset must be SITE:NAME, got '{value}'")
        names = result.setdefault(site.strip(), [])
        if name.strip() not in names:
            names.append(name.strip())
    return result or None


# Time allowed for the job to be scheduled, deployed and finished on top of the per-site task timeout.
_JOB_START_MARGIN_S = 60


def _submit_read(args) -> dict:
    """Export and submit the recipe through the shared CLI session, without downloading job results."""
    from nvflare.recipe.catalog import CatalogRecipe

    name = "catalog_read"
    recipe = CatalogRecipe(
        name=name, sites=_csv(args.sites), datasets=_dataset_map(args.datasets), task_timeout_s=args.timeout
    )
    with TemporaryDirectory(prefix="nvflare-catalog-") as job_root:
        recipe.export(job_root)
        with _catalog_session(args) as sess:
            job_id = sess.submit_job(os.path.join(job_root, name))
            data = {"study": args.study, "job_id": job_id}
            if not args.no_wait:
                print_human(f"catalog read job {job_id} submitted; waiting for completion", flush=True)
                last_status = None

                def progress(session, job_id, meta):
                    nonlocal last_status
                    status = meta.get("status")
                    if status != last_status:
                        print_human(f"{job_id}: {status}", flush=True)
                        last_status = status
                    return True

                rc, meta = sess.monitor_job_and_return_job_meta(
                    job_id, timeout=args.timeout + _JOB_START_MARGIN_S, cb=progress
                )
                if rc == MonitorReturnCode.TIMEOUT:
                    raise TimeoutError(
                        f"stopped waiting for catalog job {job_id}, which keeps running; "
                        f"check it with 'nvflare job meta {job_id} --study {args.study}'"
                    )
                status = meta.get("status") if isinstance(meta, dict) else None
                if rc != MonitorReturnCode.JOB_FINISHED or job_status_outcome(status) is None:
                    raise InternalError(f"no terminal status for catalog job {job_id}")
                data["status"] = status
            return data


def _render_read(data: dict):
    outcome = job_status_outcome(data.get("status"))
    if outcome is not None and outcome != "completed":
        output_error(
            "JOB_ABORTED" if outcome == "aborted" else "JOB_FAILED",
            exit_code=1,
            job_id=data["job_id"],
            detail=data["status"],
            data=data,
            hint=f"Inspect the job with 'nvflare job meta {data['job_id']} --study {data['study']}'.",
        )
    if is_json_mode():
        output_ok(data)
        return
    print_human(f"catalog read job {data['job_id']} submitted for study {data['study']}")
    if "status" in data:
        print_human(f"finished: {data['status']}")
    print_human(
        "see the result with: nvflare catalog show"
        + ("" if data["study"] == DEFAULT_STUDY else f" --study {data['study']}")
    )


def cmd_read(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_catalog_parsers()
    handle_schema_flag(
        _catalog_sub_cmd_parsers[CMD_CATALOG_READ],
        "nvflare catalog read",
        [
            "nvflare catalog read --study lung-cancer",
            "nvflare catalog read --sites site-a,site-b site-a:omop",
            "nvflare catalog read --study lung-cancer --no-wait --format json",
        ],
        sys.argv[1:],
        output_modes=["json"],
        streaming=False,
        mutating=True,
        idempotent=False,
        retry_token=_NO_RETRY_TOKEN_SCHEMA,
    )
    _run(lambda: _submit_read(args), _catalog_sub_cmd_parsers[CMD_CATALOG_READ], _render_read)


# ---- parsers -----------------------------------------------------------------------------------------------


def _define_catalog_subcommands(parser):
    sub = parser.add_subparsers(title="catalog subcommands", metavar="", dest="catalog_sub_cmd")

    p = sub.add_parser(CMD_CATALOG_SHOW, help="show complete catalog metadata, or a compact overview")
    p.add_argument("--study", default=DEFAULT_STUDY, help="study to show")
    p.add_argument("--site", default=None, help="only datasets of this site")
    p.add_argument("--dataset", default=None, help="only datasets with this name")
    p.add_argument("--overview", action="store_true", help="show a compact dataset summary instead of schema metadata")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _catalog_sub_cmd_parsers[CMD_CATALOG_SHOW] = p
    _catalog_handlers[CMD_CATALOG_SHOW] = cmd_show

    p = sub.add_parser(CMD_CATALOG_READ, help="read the datasets of the study's sites into the catalog")
    p.add_argument("--study", default=DEFAULT_STUDY, help="study to read")
    p.add_argument("--sites", default=None, help="comma-separated sites to read (default: every site of the study)")
    p.add_argument(
        "datasets",
        nargs="*",
        metavar="SITE:NAME",
        help="read only these datasets, each as SITE:NAME. A site with none listed reads all it shares",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=(
            "seconds to wait for each site (default: 300); the command stops waiting for the job "
            f"{_JOB_START_MARGIN_S} seconds after that, and the job keeps running"
        ),
    )
    p.add_argument(
        "--no-wait", action="store_true", help="return after submission (default: wait for the job to finish)"
    )
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _catalog_sub_cmd_parsers[CMD_CATALOG_READ] = p
    _catalog_handlers[CMD_CATALOG_READ] = cmd_read


def _ensure_catalog_parsers():
    global _catalog_root_parser
    if _catalog_sub_cmd_parsers and _catalog_root_parser is not None:
        return
    root = argparse.ArgumentParser(prog="nvflare catalog")
    _define_catalog_subcommands(root)
    _catalog_root_parser = root


def def_catalog_cli_parser(sub_cmd):
    global _catalog_root_parser
    parser = sub_cmd.add_parser("catalog", help="read and show the federated dataset catalog")
    _catalog_root_parser = parser
    _define_catalog_subcommands(parser)
    return {"catalog": parser}


def handle_catalog_cmd(args):
    sub_cmd = getattr(args, "catalog_sub_cmd", None)
    if sub_cmd is None:
        _ensure_catalog_parsers()
        _catalog_root_parser.print_help()
        return
    handler = _catalog_handlers.get(sub_cmd)
    if handler is None:
        raise CLIUnknownCmdException(f"Unknown catalog subcommand: {sub_cmd}")
    handler(args)
