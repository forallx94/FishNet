import numpy as np
import pickle
import csv
import os

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset


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
        transforms.Normalize((-0.017200625, -0.035683163, -0.10693816), (0.40440425, 0.39863086, 0.40172696)) # train 데이터에 0.5 정규화 적용 후 (R, G, B) 평균,분산
        ])

    batch_size = 64

    test_data_dir = [f'./cifar-10-python/test_batch']
    test_data = Customcifar10(test_data_dir, transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model_path = ['./pth/cnn_100.pth',
                './pth/resnet_100.pth',
                './pth/resnet_200.pth',
                './pth/fishnet_100.pth',
                './pth/fishnet_200.pth',
                ]

    models = ['CNN','ResNet','ResNet','FishNet','FishNet']

    header = ['Model', 'Class', 'Error rate (%)']

    with open('model_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for path, model in zip(model_path, models):
            model_name = os.path.basename(path).split('.')[0]
            if model == "CNN":
                from model.cnn import Net
                net = Net()

            elif model == "ResNet":
                from model.resnet import Bottleneck, ResNet
                net = ResNet(block = Bottleneck, layers=[2,2,2,2], num_classes=10)

            elif model == "FishNet":
                from model.fishnet import Bottleneck,FishNet
                net = FishNet(block = Bottleneck, layers=[2,2,2], num_classes=10)
            net.load_state_dict(torch.load(path))
            net = net.cuda()

            correct = 0
            total = 0
            
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
            with torch.no_grad():
                for data in test_dataloader:
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    # 신경망에 이미지를 통과시켜 출력을 계산합니다
                    outputs = net(images)
                    # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    _, predictions = torch.max(outputs, 1)
                    # 각 분류별로 올바른 예측 수를 모읍니다
                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1


            # Top-1 Error
            writer.writerow([model_name, 'Top-1 Error (%)', 100 - 100 * correct // total])

            # 각 분류별 정확도
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                writer.writerow([model_name, classname, accuracy])

if __name__ == '__main__':
    main()