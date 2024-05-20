#!/usr/bin/python
import xgboost as xgb
import pandas as pd

PRINT_SAMPLE = False
DATASET_ROOT = "./dataset/base_xgb_data"


def run_training() -> None:
    # Specify file path, rank 0 as the label owner, others as the feature owner
    train_path = f'{DATASET_ROOT}/train.csv'
    valid_path = f'{DATASET_ROOT}/valid.csv'

    # Load file directly to tell the match from loading with DMatrix
    df_train = pd.read_csv(train_path, header=None)
    if PRINT_SAMPLE:
        # print number of rows and columns for each worker
        print(f'Direct load: nrow={df_train.shape[0]}, ncol={df_train.shape[1]}')
        # print one sample row of the data
        print(f'Direct load: one sample row of the data: \n {df_train.iloc[0]}')

    # Load file, file will not be sharded in federated mode.
    label = "&label_column=0"
    # for Vertical XGBoost, read from csv with label_column and set data_split_mode to 1 for column mode
    dtrain = xgb.DMatrix(train_path + f"?format=csv{label}")
    dvalid = xgb.DMatrix(valid_path + f"?format=csv{label}")

    if PRINT_SAMPLE:
        # print number of rows and columns for each worker
        print(f'DMatrix: nrow={dtrain.num_row()}, ncol={dtrain.num_col()}')
        # print one sample row of the data
        data_sample = dtrain.get_data()[0]
        print(f'DMatrix: one sample row of the data: \n {data_sample}')

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

    # Save the model, only ask process 0 to save the model.
    if xgb.collective.get_rank() == 0:
        bst.save_model("./model/model.base.json")
        xgb.collective.communicator_print("Finished training\n")



if __name__ == '__main__':
    run_training()
