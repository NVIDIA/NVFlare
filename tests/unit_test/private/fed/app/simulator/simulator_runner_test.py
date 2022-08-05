import argparse
import tempfile
import unittest
import os
import shutil

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.workspace = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.workspace, 'transfer'))

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.workspace)

    def _create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("job_folder")
        parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
        parser.add_argument("--clients", "-n", type=int, help="number of clients")
        parser.add_argument("--client_list", "-c", type=str, help="client names list")
        parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)
        parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

        return parser

    # def test_something(self):
    #     workspace = tempfile.mkdtemp()
    #     os.makedirs(os.path.join(workspace, 'transfer'))
    #     parser = self._create_parser()
    #     args = parser.parse_args(['job_folder', '-o' + workspace, '-n 2', '-t 1'])
    #     runner = SimulatorRunner(args)
    #     assert runner.setup()

