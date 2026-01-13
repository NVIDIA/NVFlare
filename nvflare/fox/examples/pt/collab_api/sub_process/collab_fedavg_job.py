from nvflare.fox.examples.pt.collab_api.sub_process.collab_fedavg_train import FedAvg, Trainer
from nvflare.fox.sim import SimEnv
from nvflare.fox.sys.recipe import FoxRecipe

if __name__ == "__main__":
    print("Setting up FedAvg...")

    server = FedAvg(num_rounds=3)
    client = Trainer()

    recipe = FoxRecipe(
        job_name="fedavg",
        server=server,
        client=client,
        min_clients=2,
        inprocess=False,
        run_cmd="torchrun --nproc_per_node=2",
    )

    env = SimEnv(num_clients=2)

    print("Starting job execution...")
    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
