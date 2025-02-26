DATASET_PATH="/tmp/nvflare/dataset/creditcard.csv"
SPLIT_PATH="/tmp/nvflare/dataset/xgb_dataset/"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved CreditCard dataset in ${DATASET_PATH}"
fi

echo "Generating CreditCard data splits, reading from ${DATASET_PATH}"

echo "Split data to training/validation v.s. testing"
python3 utils/prepare_data_traintest_split.py \
--data_path "${DATASET_PATH}" \
--test_ratio 0.2 \
--out_folder "${SPLIT_PATH}"

echo "Split training/validation data"
OUTPUT_PATH="${SPLIT_PATH}/base_xgb_data"
python3 utils/prepare_data_base.py \
--data_path "${SPLIT_PATH}/train.csv" \
--out_path "${OUTPUT_PATH}"

echo "Split training/validation data vertically"
OUTPUT_PATH="${SPLIT_PATH}/vertical_xgb_data"
python3 utils/prepare_data_vertical.py \
--data_path "${SPLIT_PATH}/train.csv" \
--site_num 3 \
--out_path "${OUTPUT_PATH}"

echo "Split training/validation data horizontally"
OUTPUT_PATH="${SPLIT_PATH}/horizontal_xgb_data"
python3 utils/prepare_data_horizontal.py \
--data_path "${SPLIT_PATH}/train.csv" \
--site_num 3 \
--out_path "${OUTPUT_PATH}"
