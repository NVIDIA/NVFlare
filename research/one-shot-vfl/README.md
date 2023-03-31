# One-shot VFL

This directory will contain the code for the methods described in

### Communication-Efficient Vertical Federated Learning with Limited Overlapping Samples ([arXiv:2303.16270](https://arxiv.org/abs/2303.16270))

###### Abstract:

> Federated learning is a popular collaborative learning approach that enables clients to train a global model without sharing their local data. Vertical federated learning (VFL) deals with scenarios in which the data on clients have different feature spaces but share some overlapping samples. Existing VFL approaches suffer from high communication costs and cannot deal efficiently with limited overlapping samples commonly seen in the real world. We propose a practical vertical federated learning (VFL) framework called **one-shot VFL** that can solve the communication bottleneck and the problem of limited overlapping samples simultaneously based on semi-supervised learning. We also propose **few-shot VFL** to improve the accuracy further with just one more communication round between the server and the clients. In our proposed framework, the clients only need to communicate with the server once or only a few times. We evaluate the proposed VFL framework on both image and tabular datasets. Our methods can improve the accuracy by more than 46.5% and reduce the communication cost by more than 330× compared with state-of-the-art VFL methods when evaluated on CIFAR-10.

## License
- TBD

## Citation

> Sun, Jingwei, et al. "Communication-Efficient Vertical Federated Learning with Limited Overlapping Samples." arXiv:2303.16270. 2023.

BibTeX
```
@misc{sun2023communicationefficient,
      title={Communication-Efficient Vertical Federated Learning with Limited Overlapping Samples}, 
      author={Jingwei Sun and Ziyue Xu and Dong Yang and Vishwesh Nath and Wenqi Li and Can Zhao and Daguang Xu and Yiran Chen and Holger R. Roth},
      year={2023},
      eprint={2303.16270},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
