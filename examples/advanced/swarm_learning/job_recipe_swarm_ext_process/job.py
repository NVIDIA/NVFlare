import argparse

from torch import nn

from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe


class SmallTestModel(nn.Module):
    """Lightweight stand-in for Llama3ModelWrapper in sim/poc environments."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        return self.layers(x)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--env", type=str, default="sim", choices=["sim", "poc", "prod"])
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "small", "llama"],
        help="Model to use: 'small' for SmallTestModel, 'llama' for Llama3, "
        "'auto' selects llama for prod and small otherwise",
    )
    parser.add_argument("--model_path", type=str, default="/data")
    parser.add_argument("--command", type=str, default="python3 -u")
    parser.add_argument("--do_cse", action="store_true", default=False)
    parser.add_argument("--workspace_root", type=str, default="")
    # Prod env options
    parser.add_argument("--startup_kit_location", type=str, default="")
    parser.add_argument("--username", type=str, default="admin@nvidia.com")
    parser.add_argument("--login_timeout", type=float, default=5.0)
    return parser.parse_args()


def _create_model(args):
    use_llama = (args.model == "llama") or (args.model == "auto" and args.env == "prod")
    if use_llama:
        from pt.networks.llama3_wrapper import Llama3ModelWrapper

        return Llama3ModelWrapper(model_path=args.model_path, force_cpu=True)
    return SmallTestModel()


def _is_large_model(args):
    """Check if we're using a large model that needs extended timeouts."""
    return (args.model == "llama") or (args.model == "auto" and args.env == "prod")


def main():
    args = define_parser()
    n_clients = args.n_clients
    model = _create_model(args)
    large_model = _is_large_model(args)

    recipe_kwargs = dict(
        name="swarm-ext-process",
        model=model,
        num_rounds=args.num_rounds,
        min_clients=n_clients,
        train_script="client.py",
        launch_external_process=True,
        command=args.command,
        do_cross_site_eval=args.do_cse,
        memory_gc_rounds=1,
        cuda_empty_cache=True,
    )

    if large_model:
        # Large models (~2.8GB for Llama 1B) need extended timeouts for P2P
        # tensor streaming.  Default learn_task_ack_timeout is only 10s.
        recipe_kwargs["learn_task_ack_timeout"] = 1200
        recipe_kwargs["learn_task_timeout"] = 10800
        recipe_kwargs["final_result_ack_timeout"] = 1200

    recipe = SimpleSwarmLearningRecipe(**recipe_kwargs)

    if args.env == "prod":
        streaming_config = {
            "tensor_download_chunk_size": 2097152,
            "np_download_chunk_size": 2097152,
            "streaming_per_request_timeout": 600,
        }
        recipe.add_server_config(streaming_config)
        recipe.add_client_config(streaming_config)

    if args.env == "poc":
        from nvflare.recipe.poc_env import PocEnv

        env = PocEnv(num_clients=n_clients)
    elif args.env == "prod":
        from nvflare.recipe.prod_env import ProdEnv

        if not args.startup_kit_location:
            raise RuntimeError("--startup_kit_location is required for prod env")
        env = ProdEnv(
            startup_kit_location=args.startup_kit_location,
            login_timeout=args.login_timeout,
            username=args.username,
        )
    else:
        from nvflare.recipe.sim_env import SimEnv

        sim_kwargs = dict(num_clients=n_clients, num_threads=n_clients)
        if args.workspace_root:
            sim_kwargs["workspace_root"] = args.workspace_root
        env = SimEnv(**sim_kwargs)

    run = recipe.execute(env)
    print()
    print("Result can be found in:", run.get_result())
    print("Job Status is:", run.get_status())


if __name__ == "__main__":
    print("@@@ Starting swarm-ext-process recipe job.py")
    main()
    print("@@@ Finished swarm-ext-process recipe job.py")
