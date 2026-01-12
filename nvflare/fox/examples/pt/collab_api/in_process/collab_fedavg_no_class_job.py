"""Job Recipe: Federated Averaging with separate server and client modules.

This demonstrates how to use FoxRecipe with separate modules:
- collab_fedavg_no_class_server.py: Server-side @fox.algo
- collab_fedavg_no_class_client.py: Client-side @fox.collab

FoxRecipe auto-wraps modules with ModuleWrapper!
"""

# Import the separate server and client modules
from nvflare.fox.examples.pt.collab_api.in_process import collab_fedavg_no_class_client as client_module
from nvflare.fox.examples.pt.collab_api.in_process import collab_fedavg_no_class_server as server_module
from nvflare.fox.sim import SimEnv
from nvflare.fox.sys.recipe import FoxRecipe

if __name__ == "__main__":
    # Create recipe with separate server and client modules
    # FoxRecipe auto-wraps modules - no need for explicit ModuleWrapper!
    recipe = FoxRecipe(
        job_name="fedavg_split_modules",
        server=server_module,  # Has @fox.algo fed_avg()
        client=client_module,  # Has @fox.collab train()
        min_clients=5,
    )

    env = SimEnv(num_clients=5)
    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
