# FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction

Fishnet 모델 구현 시도  
제작 과정에서 비교군으로 두 가지 모델 설정  

CNN : [Pytorch tutorial](https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html)에서 제공된 간단한 CNN 모델  
Bottleneck ResNet18 : 기존의 [Resnet18](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L166) 의 basic block 대신 bottleneck block 을 적용  
Fishnet : 구현한 Fishnet 


| *폴더 / 파일* | *설명* |
|:-----------------:|:--------|
| cifar-10-python | 데이터셋 경로 |
| model | 모델 layer 경로 |
| pth | 모델 weight 경로 |
| train.py | 훈련 코드 |
| test.py | 테스트 코드 |
| model_result.csv | 테스트 출력 결과 |

## train

```bash
python train.py --model FishNet --save_path fishnet.pth
```

## test
```bash
python test.py
```

## Result 

모델명의 숫자는 epoch 수를 의미  
즉 fishnet_100 은 fishnet 모델 100 epochs 훈련 결과  

|    *Model*         |*Top-1 Error (%)*| *Params × 10<sup>6</sup>* | *Params* |
|:-----------------:|:--------:|:--------:|:--------:|
| [`cnn_100`](https://drive.google.com/file/d/1t6JFgDK71ioSwrySwTxgIWzc3-oASZrm/view?usp=sharing)     |   69   | 0.06 | 62006 |
| [`resnet_100`](https://drive.google.com/file/d/1OxoouWEWIofdXmCgDTrR5P5p5TYBT0U9/view?usp=sharing)  |   22   | 13 | 13962698 |
| [`resnet_200`](https://drive.google.com/file/d/1DD5JqaxSWjRFZw2yR693StlnJdYXi84w/view?usp=sharing)  |   22   | 13 | 13962698 |
| [`fishnet_100`](https://drive.google.com/file/d/1T07q-_8opYG-kbv1EZh7nkSrOgcyk4ub/view?usp=sharing) |   27   | 7 | 7998666 |
| [`fishnet_200`](https://drive.google.com/file/d/1oi2QNbEFwbwevPBldW8PeC8eDlrMKGwe/view?usp=sharing) |   26   | 7 | 7998666 |

# Reference
* [FishNet: A Versatile Backbone for Image, Region, and Pixel Level Predictio](https://arxiv.org/pdf/1901.03495.pdf)
* [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* DataLoader
    * [How to Load, Pre-process and Visualize CIFAR-10 and CIFAR -100 datasets in Python](https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html)
    * [DATASET과 DATALOADER](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)
* [Pytorch 분류기(Classifier) 학습하기](https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html)
* [Pytorch ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L108)
* [senet.pytorch](https://github.com/moskomule/senet.pytorch/blob/8cb2669fec6fa344481726f9199aa611f08c3fbd/senet/se_resnet.py#L46)
* [Image normalization](https://teddylee777.github.io/pytorch/torchvision-transform/)
* [PCA Color Augmentation](https://github.com/albumentations-team/albumentations/blob/7e49d472b451c4f25ae74d1a67503c8b3313eaf0/albumentations/augmentations/functional.py#L1091)
* [pytorch transforms 구조](https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose)