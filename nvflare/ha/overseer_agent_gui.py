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

import argparse
import os
import sys

os.environ["KIVY_NO_ARGS"] = "1"
from pprint import pformat

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.button import Button

from nvflare.ha.overseer_agent import HttpOverseerAgent


class OverseerAgentApp(App):
    def update_grid(self, app):
        sp_list = self.overseer_agent._overseer_info.get("sp_list")
        if sp_list:
            for i, sp in enumerate(sp_list):
                self.buttons[i].text = pformat(sp)
                if sp["primary"]:
                    self.buttons[i].background_color = (0.30, 1.00, 0.30, 1)
                elif sp["state"] == "offline":
                    self.buttons[i].background_color = (1.00, 0.30, 0.30, 1)
                else:
                    self.buttons[i].background_color = (0.30, 0.30, 1.00, 1)

    def on_start(self):
        self.overseer_agent = setup_basic_info()
        self.overseer_agent.start()
        self.buttons = list()
        for i in range(3):
            button = Button(text="")
            self.root.ids.splist.add_widget(button)
            self.buttons.append(button)

        Clock.schedule_interval(self.update_grid, 1)

    def on_stop(self):
        self.overseer_agent.end()


def setup_basic_info():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="example_project", help="project name")
    parser.add_argument("-r", "--role", type=str, help="role (server, client or admin)")
    parser.add_argument("-n", "--name", type=str, help="globally unique name")
    parser.add_argument("-f", "--fl_port", type=str, help="fl port number")
    parser.add_argument("-a", "--admin_port", type=str, help="adm port number")
    parser.add_argument("-s", "--sleep", type=float, help="sleep (seconds) in heartbeat")
    parser.add_argument("-c", "--ca_path", type=str, help="root CA path")
    parser.add_argument("-o", "--overseer_url", type=str, help="Overseer URL")

    args = parser.parse_args(sys.argv[1:])

    overseer_agent = HttpOverseerAgent(
        overseer_end_point=args.overseer_url,
        project=args.project,
        role=args.role,
        name=args.name,
        fl_port=args.fl_port,
        admin_port=args.admin_port,
        heartbeat_interval=args.sleep,
    )
    if args.ca_path:
        overseer_agent.set_secure_context(ca_path=args.ca_path)
    return overseer_agent


def simple_callback(overseer_agent):
    print("\nGot callback")


if __name__ == "__main__":
    OverseerAgentApp().run()
