from et_cifar10_task_processor import ETCIFAR10TaskProcessor
from model import Cifar10ConvNet

from nvflare.edge.models.model import DeviceModel
from nvflare.edge.recipes.edge_recipes import ETEdgeRecipe
from nvflare.recipe.simulation_env import SimulationExecEnv

env = SimulationExecEnv(num_clients=2)

recipe = ETEdgeRecipe(
    name="cifar10_et",
    et_model=DeviceModel(net=Cifar10ConvNet()),
    input_shape=(4, 3, 32, 32),
    output_shape=(4,),
    max_model_versions=1,
    num_updates_for_model=5,
    device_selection_size=5,
    min_hole_to_fill=5,
)
recipe.configure_simulation(task_processor=ETCIFAR10TaskProcessor("/tmp/nvflare/cifar10"), num_devices=5)
recipe.execute(env)
