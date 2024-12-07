# Adversarial Example Detection Using Robustness against Multi-Step Gaussian Noises

This repository contains the code for paper "[Adversarial Example Detection Using Robustness against Multi-Step Gaussian Noises](https://dl.acm.org/doi/10.1145/3651781.3651808)". This repository is based on [deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector) and [AE-layers](https://github.com/gmum/adversarial_examples_ae_layers/).

## Preliminaries

* [Pytorch](http://pytorch.org/): Only GPU version is available.
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)
* [scikit-image](https://scikit-image.org/)

## Pre-trained Models
We provide six pre-trained neural networks (1) three DenseNets trained on CIFAR-10, CIFAR-100 and SVHN, where models trained on CIFAR-10 and CIFAR-100 are from [odin-pytorch](https://github.com/ShiyuLiang/odin-pytroch), and (2) three ResNets trained on CIFAR-10, CIFAR-100 and SVHN. Weights of these models are in /pre_trained/.

* [DenseNet on CIFAR-10](https://www.dropbox.com/s/mqove8o9ukfn1ms/densenet_cifar10.pth?dl=0) / [DenseNet on CIFAR-100](https://www.dropbox.com/s/nosj8oblv3y8tbf/densenet_cifar100.pth?dl=0) / [DenseNet on SVHN](https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0)
* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) / [ResNet on SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0)

## Detecting Adversarial Samples

### 0. Generate adversarial samples:
```
# model: ResNet, in-distribution: CIFAR-10, adversarial attack: FGSM  gpu: 0
python ADV_Samples.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0
```

To generate samples for every attack type:
```
for model in resnet densenet
do
  for dataset in cifar10 cifar100 svhn
  do
    for adv in FGSM BIM DeepFool CWL2 PGD100
    do
      python ADV_Samples.py --dataset $dataset --net_type $model --adv_type $adv
    done
  done
done
```

### 1. Create robustness features for AE detection:
```
python create_detection_features.py  --dataset $dataset --model $model
```

To train for each combination:
```
for model in resnet densenet
do
  for dataset in cifar10 cifar100 svhn
  do
    python -u supervised_detection.py  --dataset $dataset --model $model
  done
done
```

### 2. Train AE detector with robustness features:

For supervised setting:
```
python supervised_detection.py -dataset $dataset --model $model 
```

To train for each combination:
```
for model in resnet densenet
do
  for dataset in cifar10 cifar100 svhn
  do
    python -u supervised_detection.py.py  --dataset $dataset --model $model
  done
done
```

For unsupervised setting:
```
python unsupervised_detection.py --dataset $dataset --model $model
```

To train for each combination:
```
for model in resnet densenet
do
  for dataset in cifar10 cifar100 svhn
  do
    python -u unsupervised_detection.py.py  --dataset $dataset --model $model
  done
done
```


### 3. View result tables:
```
python view_results.py
```

Tables containing the final results should be displayed:
```
Supervised
                   | BIM   | CW    | DF    | FGSM  | PGD   | AVG   | 
densenet | cifar10 | 99.98 | 96.31 | 95.25 | 99.93 | 100.00| 98.29 | 
         | cifar100| 99.97 | 98.66 | 94.88 | 100.00| 99.66 | 98.63 | 
         | svhn    | 99.79 | 98.20 | 98.20 | 98.96 | 100.00| 99.03 | 
resnet   | cifar10 | 99.46 | 95.24 | 96.06 | 99.25 | 99.76 | 97.95 | 
         | cifar100| 98.10 | 91.20 | 88.96 | 99.82 | 99.69 | 95.55 | 
         | svhn    | 98.53 | 95.93 | 97.55 | 96.52 | 99.71 | 97.65 | 
```

```
Unsupervised
                   | BIM   | CW    | DF    | FGSM  | PGD   | AVG   | 
densenet | cifar10 | 99.67 | 83.60 | 83.36 | 86.54 | 99.79 | 90.59 | 
         | cifar100| 98.46 | 58.59 | 56.41 | 70.04 | 95.13 | 75.73 | 
         | svhn    | 97.05 | 93.26 | 94.37 | 90.93 | 99.49 | 95.02 | 
resnet   | cifar10 | 90.87 | 79.03 | 92.62 | 87.10 | 94.37 | 88.80 | 
         | cifar100| 93.44 | 72.64 | 75.69 | 76.70 | 95.86 | 82.86 | 
         | svhn    | 89.29 | 87.36 | 95.75 | 90.21 | 95.30 | 91.58 | 
```

