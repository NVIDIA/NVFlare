# FedCE: Fair Federated Learning via Client Contribution Estimation

## Introduction to MONAI and FedCE

### MONAI
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) on medical image applications.
It uses [MONAI](https://github.com/Project-MONAI/MONAI),
which is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem.

### FedCE
This example illustrates the fair federated learning algorithm [FedCE](https://arxiv.org/abs/2303.16520) accepted to [CVPR2023](https://cvpr2023.thecvf.com/). 
It estimates client contribution in gradient and data spaces, and utilizes this client Contribution Estimation to guide the federated learning process towards better collaboration and performance fairness.

## License
- The code in this directory is released under Apache v2 License.
## Citation

> Jiang, Meirui, et al. "Fair Federated Medical Image Segmentation via Client Contribution Estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

BibTeX
```
@inproceedings{jiang2023fedce,
  title={Fair Federated Medical Image Segmentation via Client Contribution Estimation},
  author={Jiang, Meirui and Roth, Holger R and Li, Wenqi and Yang, Dong and Zhao, Can and Nath, Vishwesh and Xu, Daguang and Dou, Qi and Xu, Ziyue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={--},
  year={2023}
}
```
