for dataset_name in nli squad quora
do
  echo "Training on ${dataset_name}"
  python utils/train_iterative.py --dataset_name ${dataset_name}
done
