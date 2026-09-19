# -*- coding: utf-8 -*-
"""准备测试数据: imagenette2-320 验证集, 每类 25 张共 250 张.

imagenette 是 ImageNet 的 10 类子集(fastai 制作), 类别文件夹名为 WordNet ID,
可映射回真实 ImageNet 类别 index. 配合 ImageNet 预训练模型做同分布评估,
避免跨数据集类别映射带来的准确率失真(上一版 CIFAR-10 实验准确率仅 8% 的根源).

前 50 张用作方案B 阈值校准集, 后 200 张做正式测试.
"""
import os
import socket
import tarfile
import urllib.request

import torch
from PIL import Image
from torchvision import transforms

URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'

socket.setdefaulttimeout(120)  # 防止网络卡死时脚本无限挂起


def download(url: str, dest: str, tries: int = 3):
    for i in range(1, tries + 1):
        try:
            print(f'下载中 (第 {i}/{tries} 次)...')
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # 网络抖动重试
            if i == tries:
                raise
            print(f'下载失败: {e}, 2 秒后重试')

# imagenette 类别文件夹名前缀(WordNet ID) -> 真实 ImageNet 类别 index
# index 依据 ILSVRC2012 官方 synset 顺序表逐一核对
# (tensorflow/models imagenet_lsvrc_2015_synsets.txt, 行号即 index)
WNID_TO_IMAGENET = {
    'n01440764': 0,    # tench (十棘鱼)
    'n02102040': 217,  # English springer (英国史宾格犬)
    'n02979186': 482,  # cassette player (磁带机)
    'n03000684': 491,  # chain saw (链锯)
    'n03028079': 497,  # church (教堂)
    'n03394916': 566,  # French horn (圆号)
    'n03417042': 569,  # garbage truck (垃圾车)
    'n03425413': 571,  # gas pump (加油泵)
    'n03445777': 574,  # golf ball (高尔夫球)
    'n03888257': 701,  # parachute (降落伞)
}
PER_CLASS = 25  # 10 类 x 25 = 250 张


def extract(tgz, dest):
    try:
        with tarfile.open(tgz) as t:
            t.extractall(dest, filter='data')
    except TypeError:  # 旧版 Python 无 filter 参数
        with tarfile.open(tgz) as t:
            t.extractall(dest)


def main():
    os.makedirs('data', exist_ok=True)
    root = 'data/imagenette2-320'
    if not os.path.exists(root):
        tgz = 'data/imagenette2-320.tgz'
        if not os.path.exists(tgz):
            print('下载 imagenette2-320 (约 330MB, autodl 用户可先执行 '
                  'source /etc/network_turbo 加速)...')
            download(URL, tgz)
        print('解压中...')
        extract(tgz, 'data/')

    val_dir = os.path.join(root, 'val')
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    samples, labels = [], []
    for cls in sorted(os.listdir(val_dir)):
        wnid = cls.split('_')[0]
        if wnid not in WNID_TO_IMAGENET:
            raise SystemExit(
                f'未知类别文件夹 {cls}: 不属于 imagenette 标准 10 类, '
                f'请确认数据集为 fastai 的 imagenette2-320')
        in_idx = WNID_TO_IMAGENET[wnid]
        files = sorted(os.listdir(os.path.join(val_dir, cls)))[:PER_CLASS]
        for f in files:
            img = Image.open(os.path.join(val_dir, cls, f)).convert('RGB')
            samples.append(tf(img))
            labels.append(in_idx)
    torch.save({'samples': samples, 'labels': labels}, 'data/test250.pt')
    print(f'已保存 {len(samples)} 张样本到 data/test250.pt')
    print(f'标签为真实 ImageNet index: {sorted(set(labels))}')
    print('前 50 张为阈值校准集, 后 200 张为正式测试集')


if __name__ == '__main__':
    main()
