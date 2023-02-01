import os
import numpy as np
import pickle

import torch

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices = ['CNN' , 'ResNet', 'FishNet'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

class Customcifar10(Dataset):
    def __init__(self, data_dir_list , transform=None, target_transform=None):

        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        data_batch = unpickle(data_dir_list[0])
        images = data_batch[b'data']
        labels = data_batch[b'labels']
        for data_dir in data_dir_list[1:]:
            data_batch = unpickle(data_dir)
            images = np.append(images ,data_batch[b'data'], axis=0)
            labels = np.append(labels, data_batch[b'labels'], axis=0)

        images = images.reshape(len(images),3,32,32)
        self.images = images.transpose(0,2,3,1)
        self.labels = torch.LongTensor(labels)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((-0.017200625, -0.035683163, -0.10693816), (0.40440425, 0.39863086, 0.40172696)) , # train 데이터에 0.5 정규화 적용 후 (R, G, B) 평균,분산
        transforms.RandomHorizontalFlip() , # augmentaiotn horizontalflip
        ])

    batch_size = args.batch_size

    train_data_dir = [f'./cifar-10-python/data_batch_{i}' for i in range(1,6)]
    train_data = Customcifar10(train_data_dir , transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.model == "CNN":
        from model.cnn import Net
        net = Net()

    elif args.model == "ResNet":
        from model.resnet import Bottleneck, ResNet
        net = ResNet(block = Bottleneck, layers=[2,2,2,2], num_classes=10)

    elif args.model == "FishNet":
        from model.fishnet import Bottleneck,FishNet
        net = FishNet(block = Bottleneck, layers=[2,2,2], num_classes=10)

    net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    for epoch in range(args.epochs):   # 데이터셋을 수차례 반복합니다.
        scheduler.step()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = os.path.join('./pth/',args.save_path)
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()