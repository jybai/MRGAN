# Memorization Rejection GAN (MRGAN)
This repo implements the memorization rejection technique to prevent training sample memorization in GANs proposed in our paper [Reducing Training Sample Memorization in GANs by Training with Memorization Rejection](https://arxiv.org/abs/2210.12231).
The codebase is largely derived from the [StudioGAN repo](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN), which I have no ownership over. 
Please refer to the [StudioGAN repo](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) for how to run experiments.
Arguments for controlling memorization rejection can be found in `src/main.py`.

Please cite our work if you find this repo useeful
```
@misc{https://doi.org/10.48550/arxiv.2210.12231,
      doi = {10.48550/ARXIV.2210.12231},
      url = {https://arxiv.org/abs/2210.12231},
      author = {Bai, Andrew and Hsieh, Cho-Jui and Kan, Wendy and Lin, Hsuan-Tien},
      keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Reducing Training Sample Memorization in GANs by Training with Memorization Rejection},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}
}
```
