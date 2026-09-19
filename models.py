# -*- coding: utf-8 -*-
"""模型加载、切分与规模评估.

FLOPs 统计用 forward hook 对实际执行到的 Conv2d/Linear 计数(乘加 x2),
不依赖第三方库. 切分点候选: layer1/layer2/layer3 之后, 特征图分别为
256x56x56 / 512x28x28 / 1024x14x14.
"""
import os

import torch
import torch.nn as nn
from torchvision import models


def get_resnet50():
    return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)


def get_resnet18():
    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


def get_mobilenet_v3_small():
    return models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT)


def split_resnet(resnet, cut):
    """在指定 stage 之后切分: cut in {'layer1','layer2','layer3'}.
    返回 (front, back, fc). front 为端侧, back+fc 为云端."""
    front_map = {
        'layer1': [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1],
        'layer2': [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2],
        'layer3': [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2, resnet.layer3],
    }
    back_map = {
        'layer1': [resnet.layer2, resnet.layer3, resnet.layer4],
        'layer2': [resnet.layer3, resnet.layer4],
        'layer3': [resnet.layer4],
    }
    return nn.Sequential(*front_map[cut]), nn.Sequential(*back_map[cut]), resnet.fc


def feat_shape(cut):
    return {'layer1': (256, 56, 56), 'layer2': (512, 28, 28),
            'layer3': (1024, 14, 14)}[cut]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, input_shape=(1, 3, 224, 224)):
    """前向 FLOPs 近似(乘加x2). hook 只统计实际执行到的层."""
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            macs[0] += out.numel() * (m.in_channels // m.groups) * kh * kw
        elif isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules()]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for h in hs:
        h.remove()
    return macs[0] * 2


def model_profile():
    """模型规模对比 + ResNet50 各切分点分解 -> results/model_profile.txt"""
    os.makedirs('results', exist_ok=True)
    lines = ['=== 模型规模评估 ===']
    for name, ctor in [('ResNet18', get_resnet18), ('ResNet50', get_resnet50),
                       ('MobileNetV3-Small', get_mobilenet_v3_small)]:
        m = ctor()
        lines.append(f'{name}: 参数量 {count_params(m)/1e6:.2f}M, '
                     f'FLOPs@224 {count_flops(m)/1e9:.2f}G, '
                     f'参数内存 {count_params(m)*4/1024/1024:.1f} MB')

    lines += ['', '=== ResNet50 各切分点分解 (端侧=front, 云端=back+fc) ===']
    r50 = get_resnet50()
    total_f = count_flops(r50)
    total_p = count_params(r50)
    for cut in ['layer1', 'layer2', 'layer3']:
        front, _, _ = split_resnet(r50, cut)
        c, h, w = feat_shape(cut)
        f_bytes = c * h * w * 4
        edge_f = count_flops(front)
        edge_p = count_params(front)
        lines.append(
            f'切分点 {cut} 后: 特征图 {c}x{h}x{w}, raw {f_bytes/1024:.0f}KB '
            f'({f_bytes/1024/1024:.2f}MB) | 端侧 FLOPs {edge_f/1e9:.2f}G '
            f'({edge_f/total_f*100:.1f}%), 云端 {(total_f-edge_f)/1e9:.2f}G '
            f'({(total_f-edge_f)/total_f*100:.1f}%) | 端侧参数 '
            f'{edge_p/1e6:.2f}M, 云端 {(total_p-edge_p)/1e6:.2f}M')

    out = '\n'.join(lines)
    with open('results/model_profile.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print(out)


if __name__ == '__main__':
    model_profile()
