import os
import shlex
import subprocess
import sys
from threading import Thread

from nvflare.ha.overseer_agent import HttpOverseerAgent


class OALauncher:
    def __init__(self):
        self._agent_dict = dict(server=dict(), client=dict())

    def start_overseer(self):
        new_env = os.environ.copy()
        command = [sys.executable, "-m", "nvflare.ha.overseer.overseer"]

        process = subprocess.Popen(
            command,
            preexec_fn=os.setsid,
            env=new_env,
        )
        print("Starting overseer ...")
        self._overseer_process = process

    def _get_agent(self, agent_id):
        agent = self._agent_dict["server"].get(agent_id)
        if agent is not None:
            return agent
        agent = self._agent_dict["client"].get(agent_id)
        if agent is not None:
            return agent
        raise ValueError(f"{agent_id} not found in currnet agent list")

    def start_servers(self, number):
        agent_id_list = list()
        for i in range(number):
            agent_id = f"server{i:02d}"
            agent = HttpOverseerAgent(
                "server",
                "http://localhost:5000/api/v1",
                project="test_project",
                name=agent_id,
                fl_port=str(8000 + i),
                admin_port=str(8100 + i),
            )
            thread = Thread(target=agent.start, name=agent_id)
            thread.start()
            self._agent_dict["server"][agent_id] = agent
            agent_id_list.append(agent_id)
        return agent_id_list

    def start_clients(self, number):
        agent_id_list = list()
        for i in range(number):
            agent_id = f"client{i:02d}"
            agent = HttpOverseerAgent("client", "http://localhost:5000/api/v1", project="test_project", name=agent_id)
            thread = Thread(target=agent.start, name=agent_id)
            thread.start()
            self._agent_dict["client"][agent_id] = agent
            agent_id_list.append(agent_id)
        return agent_id_list

    def pause_server(self, agent_id):
        agent = self._get_agent(agent_id)
        agent.pause()

    def resume_server(self, agent_id):
        agent = self._get_agent(agent_id)
        agent.resume()

    def get_primary_sp(self, agent_id):
        agent = self._get_agent(agent_id)
        return agent.get_primary_sp()

    def _stop(self, role):
        stopped_list = list()
        for key, agent in self._agent_dict[role].items():
            agent.end()
            stopped_list.append(key)
        return stopped_list

    def stop_servers(self):
        return self._stop("server")

    def stop_clients(self):
        return self._stop("client")

    def stop_overseer(self):
        self._overseer_process.terminate()
        try:
            self._overseer_process.wait(timeout=10)
        except subprocess.TimeoutExpired as e:
            print(f"overseer failed to stop due to {e}")
