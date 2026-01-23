"""Job Recipe: Federated Averaging with separate server and client modules.

This demonstrates how to use CollabRecipe with separate modules:
- collab_fedavg_no_class_server.py: Server-side @collab.main
- collab_fedavg_no_class_client.py: Client-side @collab.publish

CollabRecipe auto-wraps modules with ModuleWrapper!
"""

# Import the separate server and client modules
from nvflare.collab.examples.pt.publish_api.in_process import collab_fedavg_no_class_client as client_module
from nvflare.collab.examples.pt.publish_api.in_process import collab_fedavg_no_class_server as server_module
from nvflare.collab.sim import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

if __name__ == "__main__":
    # Create recipe with separate server and client modules
    # CollabRecipe auto-wraps modules - no need for explicit ModuleWrapper!
    recipe = CollabRecipe(
        job_name="fedavg_split_modules",
        server=server_module,  # Has @collab.main fed_avg()
        client=client_module,  # Has @collab.publish train()
        min_clients=5,
    )

    env = SimEnv(num_clients=5)
    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
