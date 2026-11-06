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

"""Query recorded resources for a job or study."""

import sys
from contextlib import contextmanager

from nvflare.fuel.flare_api import api_spec
from nvflare.tool import cli_output, cli_schema, cli_session
from nvflare.tool.resources import resources_output

_resources_parser = None


@contextmanager
def _resources_session(args, study="default"):
    try:
        session = cli_session.new_cli_session_for_args(
            args=args, timeout=cli_output.get_validated_connect_timeout(), study=study
        )
    except ValueError as e:
        cli_output.output_error("STARTUP_KIT_MISSING", exit_code=2, detail=str(e), hint=getattr(e, "hint", None))
        raise SystemExit(2)
    try:
        yield session
    finally:
        session.close()


def handle_resources_cmd(cmd_args):
    parser = _resources_parser
    cli_schema.handle_schema_flag(
        parser,
        "nvflare resources",
        [
            "nvflare resources --job JOB_ID",
            "nvflare resources --job JOB_ID --study STUDY_NAME --site SITE_NAME",
            "nvflare resources --study STUDY_NAME",
        ],
        sys.argv[1:],
        output_modes=["json"],
        streaming=False,
        mutating=False,
        idempotent=True,
        retry_token={"supported": False},
    )

    job_id = getattr(cmd_args, "job_id", None)
    requested_study = getattr(cmd_args, "study", None)
    site = getattr(cmd_args, "site", None)
    if site and not job_id:
        cli_output.output_error("INVALID_ARGUMENT", exit_code=4, detail="--site requires --job")
        return
    if not job_id and not requested_study:
        if cli_output.is_json_mode():
            cli_output.output_error("INVALID_ARGUMENT", exit_code=4, detail="Select --job or --study.")
        else:
            parser.print_help()
        return

    study = requested_study or "default"
    try:
        with _resources_session(cmd_args, study=study) as sess:
            if job_id:
                result = sess.get_job_resources(job_id, site=site)
            else:
                result = sess.get_study_resources()
    except api_spec.JobNotFound:
        cli_output.output_error(
            "JOB_NOT_FOUND",
            job_id=job_id,
            detail=f"searched study '{study}'",
            hint="Verify the job ID and --study using nvflare job list --study <study_name>.",
        )
        return
    except api_spec.JobNotDone:
        cli_output.output_error(
            "JOB_NOT_FINALIZED", exit_code=4, job_id=job_id, detail="resource data is available after finalization"
        )
        return
    except (api_spec.AuthenticationError, api_spec.AuthorizationError):
        raise
    except api_spec.NoConnection as e:
        cli_output.output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        return
    except Exception as e:
        cli_output.output_error("RESOURCE_DATA_UNAVAILABLE", exit_code=4, job_id=job_id, detail=str(e))
        return

    if job_id:
        summary = result.get("resource_summary")
        participant = result.get("participant_summary")
        data = {"selection": {"job_id": job_id, "site": site or "all"}, "summary": summary}
        if participant is not None:
            data["participant"] = participant
        if cli_output.is_json_mode():
            cli_output.output_ok(data)
        else:
            cli_output.print_human(
                resources_output.render_job_resources(summary, participant, job_name=result.get("job_name"))
            )
    else:
        data = {"selection": {"study": study}, "summary": result}
        if cli_output.is_json_mode():
            cli_output.output_ok(data)
        else:
            cli_output.print_human(resources_output.render_study_resources(result))


def def_resources_cli_parser(sub_cmd):
    global _resources_parser
    p = sub_cmd.add_parser(
        "resources",
        help="show recorded resources for a job or study",
        description="Read finalized resource reports for a job or accessible jobs in a study.",
    )
    p.add_argument("--job", dest="job_id", type=str, help="job ID; uses the default study unless --study is set")
    p.add_argument("--study", type=str, help="study containing the job, or study to summarize (accessible jobs only)")
    p.add_argument("--site", type=str, help="show one participant's final hardware and filesystem observation")
    cli_session.add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _resources_parser = p
    return {"resources": p}
