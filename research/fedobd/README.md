# FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning

This directory introduces the NVFLARE implementation of the quantization scheme in FedOBD, which is an integer quantization scheme that can greatly reduce the size of transferred messages.

FedOBD was accepted in [IJCAI2023](https://www.ijcai.org/proceedings/2023/0394.pdf), its latest version can be found in [arXiv:2208.05174](https://arxiv.org/abs/2208.05174)

## Abstract:

> Large-scale neural networks possess considerable expressive power. They are well-suited for complex learning tasks in industrial applications. However, large-scale models pose significant challenges for training under the current Federated Learning (FL) paradigm. Existing approaches for efficient FL training often leverage model parameter dropout. However, manipulating individual model parameters is not only inefficient in meaningfully reducing the communication overhead when training large-scale FL models, but may also be detrimental to the scaling efforts and model performance as shown by recent research. To address these issues, we propose the Federated Opportunistic Block Dropout (FedOBD) approach. The key novelty is that it decomposes large-scale models into semantic blocks so that FL participants can opportunistically upload quantized blocks, which are deemed to be significant towards training the model, to the FL server for aggregation. Extensive experiments evaluating FedOBD against four state-of-the-art approaches based on multiple real-world datasets show that it reduces the overall communication overhead by more than 88% compared to the best performing baseline approach, while achieving the highest test accuracy. To the best of our knowledge, FedOBD is the first approach to perform dropout on FL models at the block level rather than at the individual parameter level.

## License

This project is open-sourced under the Apache v2 License.

## Implementation

A quantization scheme called "ADAQUANT" has been added to NVFLARE under **nvflare/app_opt/pt/quantization**, which is based on our [official implementation](https://github.com/cyyever/distributed_learning_simulator).

## Environment Setup

```bash
# Install NVFLARE and related packages
pip install -r requirements.txt
```

## Steps to run the code

Let's follow the steps in the [quantization examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/llm_hf).

### Data Preparation
```bash
cd examples/advanced/llm_hf

mkdir dataset
cd dataset
git clone https://huggingface.co/datasets/tatsu-lab/alpaca
git clone https://huggingface.co/datasets/databricks/databricks-dolly-15k
git clone https://huggingface.co/datasets/OpenAssistant/oasst1
cd ..
mkdir dataset/dolly
python ./utils/preprocess_dolly.py --training_file dataset/databricks-dolly-15k/databricks-dolly-15k.jsonl --output_dir dataset/dolly
python ./utils/preprocess_alpaca.py --training_file dataset/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet --output_dir dataset/alpaca
python ./utils/preprocess_oasst1.py --training_file dataset/oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet --validation_file dataset/oasst1/data/validation-00000-of-00001-134b8fd0c89408b6.parquet --output_dir dataset/oasst1

python3 job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_adaquant --job_dir ${PWD}/workspace/jobs/hf_sft_adaquant --train_mode SFT --quantize_mode adaquant
```

## Citation

If you use this implementation, please cite the original FedOBD paper:

```bibtex
@inproceedings{chen2022fedobd,
    title         = {{FedOBD}: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning},
    author        = {Chen, Yuanyuan and Chen, Zichen and Wu, Pengcheng and Yu, Han},
    year          = 2023,
    booktitle     = {The 32nd International Joint Conference on Artificial Intelligence},
    doi           = {10.24963/ijcai.2023/394},
}
```
