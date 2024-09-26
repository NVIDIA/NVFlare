for dataset_name in nli squad quora all
do
    echo "Evaluation on model ${dataset_name}"
    python utils/eval_model.py --model_path /tmp/embed/cen/models_single/mpnet-base-${dataset_name}/final
done

echo "Evaluation on model federated"
python utils/eval_model.py --model_path /tmp/embed/nvflare/workspace_api/site-1/models/mpnet-base-nli/global