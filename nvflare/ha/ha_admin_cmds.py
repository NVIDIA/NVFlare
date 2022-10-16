# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging

from nvflare.apis.overseer_spec import OverseerAgent
from nvflare.fuel.hci.client.api_spec import CommandContext
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.security.logging import secure_format_exception


def list_sp(args, ctx: CommandContext):
    """Lists service provider information based on the last heartbeat from the overseer."""
    overseer_agent = ctx.get_api().service_finder.overseer_agent
    return {
        "status": APIStatus.SUCCESS,
        "details": str(overseer_agent.overseer_info),
        "data": overseer_agent.overseer_info,
    }


def get_active_sp(args, ctx: CommandContext):
    overseer_agent = ctx.get_api().service_finder.overseer_agent
    return {"status": APIStatus.SUCCESS, "details": str(overseer_agent.get_primary_sp())}


def promote_sp(args, ctx: CommandContext):
    overseer_agent = ctx.get_api().service_finder.overseer_agent
    if len(args) != 2:
        return {"status": APIStatus.ERROR_SYNTAX, "details": "usage: promote_sp example1.com:8002:8003"}

    sp_end_point = args[1]
    resp = overseer_agent.promote_sp(sp_end_point)
    if json.loads(resp.text).get("Error"):
        return {
            "status": APIStatus.ERROR_RUNTIME,
            "details": "Error: {}".format(json.loads(resp.text).get("Error")),
        }
    else:
        return {
            "status": APIStatus.SUCCESS,
            "details": "Promoted endpoint: {}. Synchronizing with overseer...".format(sp_end_point),
        }


def shutdown_system(args, ctx: CommandContext):
    api = ctx.get_api()
    overseer_agent = api.service_finder.overseer_agent
    try:
        status = api.do_command("check_status server").get("data")
        if status[0].get("data") != "Engine status: stopped":
            return {
                "status": APIStatus.ERROR_RUNTIME,
                "details": "Error: There are still jobs running. Please let them finish or abort_job before attempting shutdown.",
            }
    except Exception as e:
        return {
            "status": APIStatus.ERROR_RUNTIME,
            "details": f"Error getting server status to make sure all jobs are stopped before shutting down system: {secure_format_exception(e)}",
        }
    print("Shutting down the system...")
    resp = overseer_agent.set_state("shutdown")
    if json.loads(resp.text).get("Error"):
        return {
            "status": APIStatus.ERROR_RUNTIME,
            "details": "Error: {}".format(json.loads(resp.text).get("Error")),
        }
    else:
        return {"status": APIStatus.SUCCESS, "details": "Set state to shutdown in overseer."}


class HACommandModule(CommandModule):
    """Command module with commands for management in relation to the high availability framework."""

    def __init__(self, overseer_agent: OverseerAgent):
        self.overseer_agent = overseer_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_spec(self):
        return CommandModuleSpec(
            name="ha_mgmt",
            cmd_specs=[
                CommandSpec(
                    name="list_sp",
                    description="list service providers information from previous heartbeat",
                    usage="list_sp",
                    handler_func=list_sp,
                ),
                CommandSpec(
                    name="get_active_sp",
                    description="get active service provider",
                    usage="get_active_sp",
                    handler_func=get_active_sp,
                ),
                CommandSpec(
                    name="promote_sp",
                    description="promote active service provider to specified",
                    usage="promote_sp sp_end_point",
                    handler_func=promote_sp,
                ),
                CommandSpec(
                    name="shutdown_system",
                    description="shut down entire system by setting the system state to shutdown through the overseer",
                    usage="shutdown_system",
                    handler_func=shutdown_system,
                ),
            ],
        )

    def list_sp(self, args, api):
        """List service provider information based on the last heartbeat from the overseer.

        Details are used for displaying the response in the CLI, and data is the data in a dict that is provided in FLAdminAPI.

        """
        return {
            "status": APIStatus.SUCCESS,
            "details": str(self.overseer_agent._overseer_info),
            "data": self.overseer_agent._overseer_info,
        }

    def get_active_sp(self, args, api):
        return {"status": APIStatus.SUCCESS, "details": str(self.overseer_agent.get_primary_sp())}

    def promote_sp(self, args, api):
        if len(args) != 2:
            return {"status": APIStatus.ERROR_SYNTAX, "details": "usage: promote_sp example1.com:8002:8003"}

        sp_end_point = args[1]
        resp = self.overseer_agent.promote_sp(sp_end_point)
        if json.loads(resp.text).get("Error"):
            return {
                "status": APIStatus.ERROR_RUNTIME,
                "details": "Error: {}".format(json.loads(resp.text).get("Error")),
            }
        else:
            return {
                "status": APIStatus.SUCCESS,
                "details": "Promoted endpoint: {}. Synchronizing with overseer...".format(sp_end_point),
            }

    def shutdown_system(self, args, api):
        try:
            status = api.do_command("check_status server").get("data")
            if status[0].get("data") != "Engine status: stopped":
                return {
                    "status": APIStatus.ERROR_RUNTIME,
                    "details": "Error: There are still jobs running. Please let them finish or abort_job before attempting shutdown.",
                }
        except Exception as e:
            return {
                "status": APIStatus.ERROR_RUNTIME,
                "details": "Error getting server status to make sure all jobs are stopped before shutting down system: {}".format(
                    secure_format_exception(e)
                ),
            }
        print("Shutting down the system...")
        resp = self.overseer_agent.set_state("shutdown")
        if json.loads(resp.text).get("Error"):
            return {
                "status": APIStatus.ERROR_RUNTIME,
                "details": "Error: {}".format(json.loads(resp.text).get("Error")),
            }
        else:
            return {"status": APIStatus.SUCCESS, "details": "Set state to shutdown in overseer."}
