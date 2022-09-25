.. _fl_simulator:

#########################
NVIDIA FLARE FL Simulator
#########################

The NVIDIA FLARE FL Simulator was added in version 2.2 to help researchers accelerate the development of federated
learning workflows.

The FL Simulator is a lightweight simulator of a running NVFLARE FL deployment, and it can allow researchers to test
and debug their application without provisioning a real project. The FL jobs run on a server and 
multiple clients in the same process but in a similar way to how it would run in a real deployment so researchers
can more quickly build out new components and jobs that can then be directly used in a real production deployment.

***********************
Command Usage
***********************

.. code-block::

    usage: nvflare simulator [-h] -w WORKSPACE [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] job_folder

    positional arguments:
    job_folder

    optional arguments:
    -h, --help            show this help message and exit
    -w WORKSPACE, --workspace WORKSPACE
                            WORKSPACE folder
    -n N_CLIENTS, --n_clients N_CLIENTS
                            number of clients
    -c CLIENTS, --clients CLIENTS
                            client names list
    -t THREADS, --threads THREADS
                            number of parallel running clients
    -gpu GPU, --gpu GPU   list of GPU Device Ids, comma separated

*****************
Command examples
*****************

1. Run the single NVFlare app:

This command will run the same app on the server and 8 clients using 1 thread. The client names will be site-1, site-2, ... , site-8:

.. code-block::python

    python3 -u -m nvflare.private.fed.app.simulator.simulator app_folder -o WORKSPACE_FOLDER -n 8 -t 1


2. Run the NVFlare job 

This command will run the job (in the job_folder), following the job meta.json. The executing client list is provided in the command line ("client0,client1,client2,client3"). If there is any client not defined in the deploy_map of the meta.json, the simulator will report error and not run.

.. code-block::python

    python3 -u -m nvflare.private.fed.app.simulator.simulator job_folder -o WORKSPACE_FOLDER -c client0,client1,client2,client3 -t 1


3. Not providing client name list

If there is no client name list provided, and there's no number of clients option provided, the simulator is going to extract the client names list from the deployment_map in the meta.json to run.

.. code-block::python

    python3 -u -m nvflare.private.fed.app.simulator.simulator job_folder -o WORKSPACE_FOLDER  -t 1


.. note::

    The client name list option is used in priority over the number of clients option. When it's provided, it will be used as the simulated client name list.

**************************
Debug NVFlare Application
**************************

One of the goals for the Simulator is to enable researchers easily debug the NVFlare application. The FL simulator is implemented in a way of API design. Actually, the Simulator application is also implemented using the Simulator API. The researchers can simply write a "main" python script like the Simulator App, then place the script into their familiar Python IDE, add the NVFlare app into the python source codes path, then add the breakpoints to debug the application run.

.. code-block::python

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("job_folder")
        # parser.add_argument("--data_path", "-i", type=str, help="Input data_path")
        parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
        parser.add_argument("--clients", "-n", type=int, help="number of clients")
        parser.add_argument("--client_list", "-c", type=str, help="client names list")
        parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)
        parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
        args = parser.parse_args()
        return args
    
    
    if __name__ == "__main__":
        """
        This is the main program when starting the NVIDIA FLARE server process.
        """
    
        if sys.version_info >= (3, 9):
            raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
        if sys.version_info < (3, 7):
            raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")
        args = parse_args()
    
        simulator = SimulatorRunner(args)
        if simulator.setup():
            simulator.run()
        os._exit(0)

***************************
SWAP_IN and SWAP_OUT events
***************************
During the FLARE simulator execution, the client Apps are executed in turn in the same execution thread. Each executing client App will go fetching the task from the controller on the server, executing the task, and then submitting the task results to the controller. Once submitting results finished, the current client App will yield the executing thread to the next client App to execute. If the client App needs to preserve some states for the next "execution turn" to continue, the client executor can make use of the "SWAP_OUT" event fired by the simulator engine to save the current states. When the client App gets the turn to execute again, then use the "SWAP_IN" event to recover the previous saved states.

****************************************************
Multi-GPU and Separate Client Process with Simulator
****************************************************
The simulator "-t" option provides the option to use how many threads to run the simulator. The simulator runs within the same process, and it makes use of the single GPU. If there's multiple GPUs available and you want to make use of them all for the simulator run, you can use the "-gpu" option for this. The "-gpu" option provides the "," list of GPUs for the simulator to run on. The clients list will be distributed among the GPUs.

For example: 

.. code-block::shell

  -c  c1,c2,c3,c4,c5 -gpu 0,1

The client c1, c3, c5 will run on GPU 0 in one process, and client c2, c4 will run on GPU 1 in another process.

The GPU numbers do not have to be unique. If you use "-gpu 0,0", this will run 2 separate client processes on GPU 0, assuming this GPU will have enough memory to support the applicationss.
