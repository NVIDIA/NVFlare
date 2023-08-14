# ConDistFL: Conditional Distillation for Federated Learning from Partially Annotated Data

This folder will contain code to run the experiments reported in 

### ConDistFL: Conditional Distillation for Federated Learning from Partially Annotated Data ([arXiv:2308.04070](https://arxiv.org/abs/2308.04070))
Accepted to the 4th Workshop on Distributed, Collaborative, & Federated Learning ([DeCaF](https://decaf-workshop.github.io/decaf-2023/)), Vancouver, October 12th, 2023.

###### Abstract:
> Developing a generalized segmentation model capable of simultaneously delineating multiple organs and diseases is highly desirable. Federated learning (FL) is a key technology enabling the collaborative development of a model without exchanging training data. However, the limited access to fully annotated training data poses a major challenge to training generalizable models. We propose "ConDistFL", a framework to solve this problem by combining FL with knowledge distillation. Local models can extract the knowledge of unlabeled organs and tumors from partially annotated data from the global model with an adequately designed conditional probability representation. We validate our framework on four distinct partially annotated abdominal CT datasets from the MSD and KiTS19 challenges. The experimental results show that the proposed framework significantly outperforms FedAvg and FedOpt baselines. Moreover, the performance on an external test dataset demonstrates superior generalizability compared to models trained on each dataset separately. Our ablation study suggests that ConDistFL can perform well without frequent aggregation, reducing the communication cost of FL.

## License
TBD

## Citation

> Wang, Pochuan, et al. "ConDistFL: Conditional Distillation for Federated Learning from Partially Annotated Data." arXiv preprint arXiv:2308.04070 (2023).

BibTeX
```
@article{wang2023condistfl,
  title={ConDistFL: Conditional Distillation for Federated Learning from Partially Annotated Data},
  author={Wang, Pochuan and Shen, Chen and Wang, Weichung and Oda, Masahiro and Fuh, Chiou-Shann and Mori, Kensaku and Roth, Holger R},
  journal={arXiv preprint arXiv:2308.04070},
  year={2023}
}
```
