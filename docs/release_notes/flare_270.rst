**************************
What's New in FLARE v2.7.0
**************************

The new features can be divided into three categories

FLARE Core
==========

Job Recipe (Technical Preview)
------------------------------
  Introducing new **Flare Job Recipe**: Simple Recipe to capture the code needed to specify the client training and server algorithm. This should greatly
  simplify the data scientists code to write for federated learning job. The same Job Recipe can be run in SimEnv, PoCEnv, ProdEnv.

  > Note: this feature is technical review, as we haven't convert all the example and code to Job Recipe.
  > But more than half-dozen recipes are provided for you to use.

  Here is an example of the FedAvg Job Recipe

  ```
    n_clients = 2
    num_rounds = 3
    train_script = "client.py"

    recipe = FedAvgRecipe(
        name="hello-tf_fedavg",
        num_rounds=num_rounds,
        initial_model=Net(),
        min_clients=n_clients,
        train_script=train_script,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env=env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()

  ```

you can find more about the recipe in the [Job Recipe Tutorials]( <add tutorial links>) and  [documentation](<add link>)

Memory Management Improvements
------------------------------



Security Enhancement
--------------------
-- Unsafe Deserialization - torch.jit.load  is replaced with safe-tensor based implementation

-- Unsafe Deserialization - Function Call -- FOB auto-registration is removed. A white listed FOBs are auto-registered.

-- Command Injection via Grep Parameters -- commands are reimplemented to avoid command injections


Networking Support
------------------



Pre-Install CLI command
--------------------------------



Confidential Federated AI
=========================


FLARE Edge
==========

