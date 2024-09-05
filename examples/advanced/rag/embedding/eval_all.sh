for dataset_name in nli squad quora all
do
    echo "Evaluation on ${dataset_name} with model"
    python utils/eval_model.py --model_path /tmp/embed/cen/models_single/mpnet-base-${dataset_name}/final
done

python utils/eval_model.py --model_path /tmp/embed/nvflare/workspace/site-1/simulate_job/app_site-1/models/mpnet-base-nli/global