# Transfer-for-DVS in Pytorch
Here is the PyTorch implementation of our paper.

**Paper Title: "An Efficient Knowledge Transfer Strategy for Spiking Neural Networks from Static to Event Domain"**

**Authors: Xiang He\*, Dongcheng Zhao\*, Yang Li\*, Guobin Shen, Qingqun Kong, Yi Zeng**

**Accepted by: The 38th Annual AAAI Conference on Artificial Intelligence (AAAI 2024, Oral Presentation)**

\[[arxiv](https://arxiv.org/abs/2303.13077)\] \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/27806)\] \[[code](https://github.com/Brain-Cog-Lab/Transfer-for-DVS)\]

## Why we need a transfer

• Event-based datasets are usually less annotated, and the small data scale makes SNNs prone to overfitting

• While static images intuitively provide rich spatial information that may benefit event data, exploiting this knowledge remains a difficult problem. This is because that static and event data represent different modalities with domain mismatch.

<img src="fig\DomainMismatch.jpg" style="zoom: 50%;" />



## Method Introduction

•**To address this problem, we propose solutions in terms of two aspects:** **feature distribution** **and** **training strategy.**

1. **Knowledge Transfer Loss**

   •**Learn spatial domain-invariant features and provides dynamically learnable coefficients**

   •**Reduce domain distribution differences**

2. **Sliding Training**

   •**Static image inputs are gradually replaced with event data probabilistically during training process**

   •**Result in a smoother and more stable learning process.**

<img src="fig\our_method.jpg" style="zoom: 67%;" />


## NCALTECH101 & CEP-DVS Datasets Prepare

We provide download links to the NCALTECH101 and CEP-DVS datasets used in the paper, which you can find in the reply below this issue!
[Ncaltech的使用](https://github.com/Brain-Cog-Lab/Transfer-for-DVS/issues/2#issuecomment-2266674665)

We use the tonic library for event data reading. Specifically, we used version 0.0.1 of the tonic library, which is no longer available on the official website, and we uploaded the tonic libraries that the project depends on to a network drive.
You can download it from [here](https://pan.baidu.com/s/1LCimoFgbfAweYu-uJ-WyUA), and the extraction code is 3x4r.

Please put it in the tonic environment at a location such as: `/home/anaconda3/envs/all_hx/lib/python3.8/site-packages/tonic/`.

## Usage

The well-trained model can be found at [here](https://huggingface.co/xianghe/transfer_for_dvs/tree/main).

If you want to retrain yourself to verify the results in the paper, please refer to the commands in scripts [run_aba.sh](./run_aba.sh) and [run_omni.sh](./run_omni.sh). 

As an example, the script for using our method on the N-Caltech101 dataset would look like this:

```shell
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --regularization
```


## Citation
If our paper is useful for your research, please consider citing it:
```latex
@inproceedings{he2024efficient,
  title={An Efficient Knowledge Transfer Strategy for Spiking Neural Networks from Static to Event Domain},
  author={He, Xiang and Zhao, Dongcheng and Li, Yang and Shen, Guobin and Kong, Qingqun and Zeng, Yi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={512--520},
  year={2024}
}
```

## Acknowledgements

This code began with [Brain-Cog](https://github.com/BrainCog-X/Brain-Cog),  and the code for the visualization is from [https://github.com/tomgoldstein/loss-landscape ](https://github.com/tomgoldstein/loss-landscape ) and [https://github.com/jacobgil/pytorch-grad-cam ](https://github.com/jacobgil/pytorch-grad-cam ). Thanks for their great work. If you are confused about using it or have other feedback and comments, please feel free to contact us via hexiang2021@ia.ac.cn. Have a good day!
