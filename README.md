This repository includes the evaluation code for the paper "**Regularizing Proxies with Multi-Adversarial Training for Unsupervised Domain-Adaptive Semantic Segmentation** ([Arxiv Link](https://arxiv.org/pdf/1907.12282.pdf))". The whole package will be made public soom.

## Dependencies:
mxnet

gluoncv

numpy

tqdm

easydict

yaml

pillow


## Dataset:
To evaluate the performance on Cityscapes dataset, please first put the dataset into the correct path. Please edit the variable "data_root" in "cfg/resnet101_gta2cs.yaml" and "cfg/resnet101_syn2cs.yaml", which points to the data root. Then name the Cityscapes folder "cityscapes" and put it in the data root. 

## Evaluation:
We provide two resnet101 models for GTAV->CS and SYNTHIA->CS respectively. first download the models [HERE](https://1drv.ms/u/s!Akvzt7Vno--Za-wbJG3sYFT4_7Q?e=sNmqkQ) and [HERE](https://1drv.ms/u/s!Akvzt7Vno--ZarB0vDBR6FFGLNg?e=BhXvgv):

To evaluate them, simply run:

```python eval.py --cfg cfg/resnet101_gta2cs.yaml --resume resnet101_gta2cs.params```

```python eval.py --cfg cfg/resnet101_syn2cs.yaml --resume resnet101_syn2cs.params```