# FL Gradient Inversion

This directory will contain the tools necessary to recreate the chest X-ray 
experiments described in


### Do Gradient Inversion Attacks Make Federated Learning Unsafe? [arXiv:2202.06924](https://arxiv.org/abs/2202.06924)
Accepted to [IEEE Transactions on Medical Imaging](https://www.embs.org/tmi)

###### Abstract:

> Federated learning (FL) allows the collaborative training of AI models without needing to share raw data. This capability makes it especially interesting for healthcare applications where patient and data privacy is of utmost concern. However, recent works on the inversion of deep neural networks from model gradients raised concerns about the security of FL in preventing the leakage of training data. In this work, we show that these attacks presented in the literature are impractical in real FL use-cases and provide a new baseline attack that works for more realistic scenarios where the clients' training involves updating the Batch Normalization (BN) statistics. Furthermore, we present new ways to measure and visualize potential data leakage in FL. Our work is a step towards establishing reproducible methods of measuring data leakage in FL and could help determine the optimal tradeoffs between privacy-preserving techniques, such as differential privacy, and model accuracy based on quantifiable metrics.

## Citation

> Hatamizadeh A, Yin H, Molchanov P, Myronenko A, Li W, Dogra P, Feng A, Flores MG, Kautz J, Xu D, Roth HR. Do Gradient Inversion Attacks Make Federated Learning Unsafe?. arXiv preprint arXiv:2202.06924. 2022 Feb 14.

BibTeX
```
@article{hatamizadeh2022gradient,
  title={Do Gradient Inversion Attacks Make Federated Learning Unsafe?},
  author={Hatamizadeh, Ali and Yin, Hongxu and Molchanov, Pavlo and Myronenko, Andriy and Li, Wenqi and Dogra, Prerna and Feng, Andrew and Flores, Mona G and Kautz, Jan and Xu, Daguang and others},
  journal={arXiv preprint arXiv:2202.06924},
  year={2022}
}
```
