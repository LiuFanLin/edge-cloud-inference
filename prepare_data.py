# -*- coding: utf-8 -*-
"""准备测试数据：下载 CIFAR-10 测试集，取 100 张，resize 到 224x224"""
import os
import torch
from torchvision import datasets, transforms

os.makedirs('data', exist_ok=True)

print('下载 CIFAR-10 测试集...')
test_set = datasets.CIFAR10(root='./data', train=False, download=True)

# CIFAR-10 -> ImageNet 类别映射
# CIFAR-10 类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# 映射到 ImageNet 对应类别的 index
CIFAR_TO_IMAGENET = {
    0: 404,   # airplane -> airliner
    1: 511,   # automobile -> convertible
    2: 10,    # bird -> brambling (bird)
    3: 281,   # cat -> tabby cat
    4: 352,   # deer -> impala
    5: 207,   # dog -> golden retriever
    6: 30,    # frog -> bullfrog
    7: 339,   # horse -> zebra (closest)
    8: 510,   # ship -> container ship
    9: 555,   # truck -> fire engine
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

samples = []
labels = []
imagenet_labels = []
for i in range(100):
    img, label = test_set[i]
    tensor = transform(img)
    samples.append(tensor)
    labels.append(label)
    imagenet_labels.append(CIFAR_TO_IMAGENET[label])

torch.save({
    'samples': samples,
    'labels': labels,
    'imagenet_labels': imagenet_labels,
    'class_names': test_set.classes,
    'cifar_to_imagenet': CIFAR_TO_IMAGENET,
}, 'data/cifar100.pt')

print(f'已保存 100 张样本到 data/cifar100.pt')
print(f'CIFAR-10 类别: {test_set.classes}')
print(f'映射到 ImageNet 类别 index: {list(CIFAR_TO_IMAGENET.values())}')
