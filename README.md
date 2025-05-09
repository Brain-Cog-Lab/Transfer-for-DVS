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


## NCALTECH101 & CEP-DVS Datasets Preparation

We provide download links to the NCALTECH101 and CEP-DVS datasets used in the paper, which you can find in the reply below this issue!
[Ncaltech的使用](https://github.com/Brain-Cog-Lab/Transfer-for-DVS/issues/2#issuecomment-2266674665)

We use the tonic library for event data reading. Specifically, we used version 0.0.1 of the tonic library, which is no longer available on the official website, and we uploaded the tonic libraries that the project depends on to a network drive.
You can download it from [here](https://pan.baidu.com/s/1LCimoFgbfAweYu-uJ-WyUA), and the extraction code is 3x4r.

Please put it in the tonic environment at a location such as: `/home/anaconda3/envs/all_hx/lib/python3.8/site-packages/tonic/`.

## Usage

#### Validation

If you would like to verify the results in the paper, The well-trained model can be found at [here](https://huggingface.co/xianghe/transfer_for_dvs/tree/main).

As an example, you could run the following code and get 92.64% accuracy on Ncaltech101.

```
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --regularization --eval_checkpoint /home/hexiang/DomainAdaptation_DVS/Results2/train_DomainAdaptation/Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_42-DA_False-ls_0.0-domainLoss_True_coefficient0.5-traindataratio_1.0-rgbdataratio_1.0-TET_loss_True-hsv_True-sl_True-regularization_True/model_best.pth.tar --eval
```

<img src="fig\validation_results.jpg" style="zoom: 90%;" />


#### Training


If you want to retrain yourself to verify the results in the paper, please refer to the commands in scripts [run_aba.sh](./run_aba.sh) and [run_omni.sh](./run_omni.sh). 

As an example, the script for using our method on the N-Caltech101 dataset would look like this:

```shell
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --domain-loss-coefficient 0.5 --TET-loss --regularization
```

With step=10, batch_size=120, on a single 40G A100 GPU, it takes `38000MB` of gpu memory. running time is about `4.5` hours for 300 epochs. 

You can adjust the batch size yourself. This will not affect performance and may even give better results.


#### Loss landscape Visualization

An example of a script to visualize a loss landscape plot is:

```shell
python main_visual_losslandscape.py --model resnet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 512 --eval --eval_checkpoint /home/hexiang/DomainAdaptation_DVS/Results2/train_DomainAdaptation/Transfer_ResNet18-CEPDVS-6-bs_120-seed_42-DA_True-ls_0.0-domainLoss_True_coefficient0.5-traindataratio_1.0-rgbdataratio_1.0-TET_loss_True-hsv_True-sl_True-regularization_True/model_best.pth.tar --x=-1.0:1.0:51 --y=-1.0:1.0:51 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --DVS-DA --smoothing 0.0 --output /home/hexiang/DomainAdaptation_DVS/Results2/ --train-portion 0.5 --num-classes 20
```

#### More discussion
In this paper, we only validate the results on SNNs, and we believe that this approach is not limited by the network structure, i.e., the proposed efficient knowledge migration method should simultaneously contribute to the performance of the ANN model.



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
