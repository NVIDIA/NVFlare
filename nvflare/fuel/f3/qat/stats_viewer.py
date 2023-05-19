# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import cmd
import json

from nvflare.fuel.f3.stats_pool import VALID_HIST_MODES, StatsPoolManager, parse_hist_mode
from nvflare.fuel.hci.table import Table


class StatsViewer(cmd.Cmd):
    def __init__(self, pools: dict, prompt: str = "> "):
        cmd.Cmd.__init__(self)
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = prompt
        self.pools = pools
        StatsPoolManager.from_dict(pools)

    def do_list_pools(self, arg):
        headers, rows = StatsPoolManager.get_table()
        self._show_table(headers, rows)

    def do_show_pool(self, arg: str):
        args = arg.split()
        if len(args) < 1:
            self.write_string("Error: missing pool name")
            return
        name = args[0]
        mode = ""
        if len(args) > 1:
            mode = args[1]
        mode = parse_hist_mode(mode)
        if not mode:
            self.write_string(f"Error: invalid model {args[1]} - must be one of {VALID_HIST_MODES}")
            return
        pool = StatsPoolManager.get_pool(name)
        if not pool:
            self.write_string(f"Error: pool '{name}' does not exist")
            return

        headers, rows = pool.get_table(mode)
        self._show_table(headers, rows)

    def _show_table(self, headers, rows):
        t = Table(headers)
        for r in rows:
            t.add_row(r)
        t.write(self.stdout)

    def do_bye(self, arg):
        return True

    def emptyline(self):
        return

    def run(self):
        self.cmdloop(self.intro)

    def _write(self, content: str):
        self.stdout.write(content)

    def write_string(self, data: str):
        content = data + "\n"
        self._write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_file", "-f", type=str, help="stats file name", required=True)
    args = parser.parse_args()

    with open(args.stats_file) as f:
        d = json.load(f)

    viewer = StatsViewer(d)
    viewer.run()


if __name__ == "__main__":
    main()
