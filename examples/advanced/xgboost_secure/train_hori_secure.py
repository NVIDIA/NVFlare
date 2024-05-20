#!/usr/bin/python
import multiprocessing
import sys
import time

import xgboost as xgb
import xgboost.federated
import pandas as pd

PRINT_SAMPLE = False
DATASET_ROOT = "./dataset/horizontal_xgb_data"

def run_server(port: int, world_size: int) -> None:
    xgboost.federated.run_federated_server(port, world_size)


def run_worker(port: int, world_size: int, rank: int) -> None:
    communicator_env = {
        'xgboost_communicator': 'federated',
        'federated_server_address': f'localhost:{port}',
        'federated_world_size': world_size,
        'federated_rank': rank,
        'plugin_name': 'mock',
        'loader_params': {'LIBRARY_PATH': '/tmp'},
        'proc_params': {'': ''},
    }

    # Always call this before using distributed module
    with xgb.collective.CommunicatorContext(**communicator_env):
        # Specify file path, rank 0 as the label owner, others as the feature owner
        train_path = f'{DATASET_ROOT}/site-{rank + 1}/train.csv'
        valid_path = f'{DATASET_ROOT}/site-{rank + 1}/valid.csv'

        # Load file directly to tell the match from loading with DMatrix
        df_train = pd.read_csv(train_path, header=None)
        if PRINT_SAMPLE:
            # print number of rows and columns for each worker
            print(f'Direct load: rank={rank}, nrow={df_train.shape[0]}, ncol={df_train.shape[1]}')
            # print one sample row of the data
            print(f'Direct load: rank={rank}, one sample row of the data: \n {df_train.iloc[0]}')

        # Load file, file will not be sharded in federated mode.
        label = "&label_column=0"
        # for Vertical XGBoost, read from csv with label_column and set data_split_mode to 1 for column mode
        dtrain = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=3)
        dvalid = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=3)

        if PRINT_SAMPLE:
            # print number of rows and columns for each worker
            print(f'DMatrix: rank={rank}, nrow={dtrain.num_row()}, ncol={dtrain.num_col()}')
            # print one sample row of the data
            data_sample = dtrain.get_data()[0]
            print(f'DMatrix: rank={rank}, one sample row of the data: \n {data_sample}')

        # Specify parameters via map, definition are same as c++ version
        param = {
            "max_depth": 3,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "nthread": 1,
        }

        # Specify validations set to watch performance
        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
        num_round = 3

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist)

        # Save the model, same model for all ranksonly ask process 0 to save the model.
        if xgb.collective.get_rank() == 0:
            bst.save_model("./model/model.hori.secure.json")
            xgb.collective.communicator_print("Finished training\n")


def run_federated() -> None:
    port = 5555
    world_size = int(sys.argv[1])

    server = multiprocessing.Process(target=run_server, args=(port, world_size))
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(target=run_worker,
                                         args=(port, world_size, rank))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    server.terminate()


if __name__ == '__main__':
    run_federated()
