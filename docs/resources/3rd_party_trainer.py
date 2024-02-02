import argparse
import logging

from nvflare.client.flare_agent import AgentClosed, FlareAgentWithCellPipe

NUMPY_KEY = "numpy_key"


def main():

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="workspace folder", required=False, default=".")
    parser.add_argument("--site_name", "-s", type=str, help="flare site name", required=True)
    parser.add_argument("--agent_id", "-a", type=str, help="agent id", required=True)

    args = parser.parse_args()

    # 1. create the agent
    agent = FlareAgentWithCellPipe(
        root_url="grpc://server:8002",
        site_name=args.site_name,
        agent_id=args.agent_id,
        workspace_dir=args.workspace,
        secure_mode=True,
        submit_result_timeout=2.0,
        heartbeat_timeout=120.0,
    )

    # 2. start the agent
    agent.start()

    # 3. processing tasks
    while True:
        print("getting task ...")
        try:
            task = agent.get_task()
        except AgentClosed:
            print("agent closed - exit")
            break

        print(f"got task: {task}")
        result = train(task.data)  # perform train task
        submitted = agent.submit_result(result)
        print(f"result submitted: {submitted}")

    # 4. stop the agent
    agent.stop()


def train(model):
    print(f"training on {model}")
    return model


if __name__ == "__main__":
    main()
