# -*- coding: utf-8 -*-
"""端侧客户端：实现两种协同方案"""
import torch
import io
import numpy as np
import requests
from PIL import Image
from models import get_resnet18, split_resnet18
from torchvision import transforms

CLOUD = 'http://localhost:5000'

print('加载端侧模型...')
resnet_full = get_resnet18()
resnet_full.eval()
front, _, _ = split_resnet18(resnet_full)
front.eval()

CONFIDENCE_THRESHOLD = 0.8


def run_split(img_tensor):
    """方案A: 分割推理
    端侧: 跑前半 -> 输出 128x28x28 特征图
    云端: 跑后半 -> 输出预测
    每次都上传特征图
    返回: (预测类别, 通信字节数, 端侧耗时ms, 云端耗时ms)
    """
    # 端侧计算
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        feat = front(img_tensor.unsqueeze(0))
    t1 = time.perf_counter()
    edge_ms = (t1 - t0) * 1000

    # 传输 + 云端
    payload = feat.numpy().tobytes()
    t2 = time.perf_counter()
    resp = requests.post(f'{CLOUD}/split', data=payload)
    t3 = time.perf_counter()
    cloud_ms = (t3 - t2) * 1000

    return int(resp.text), len(payload), edge_ms, cloud_ms


def run_offload(img_tensor):
    """方案B: 置信度驱动卸载
    端侧: 完整 ResNet18 跑出预测和置信度
    置信度 >= 0.8: 直接用端侧结果, 不上传
    置信度 < 0.8:  上传原图给云端重判
    返回: (预测类别, 通信字节数, 端侧耗时ms, 云端耗时ms, 是否上传)
    """
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        out = resnet_full(img_tensor.unsqueeze(0))
        prob = torch.softmax(out, dim=1)
        conf, pred = prob.max(dim=1)
    t1 = time.perf_counter()
    edge_ms = (t1 - t0) * 1000

    if conf.item() >= CONFIDENCE_THRESHOLD:
        return pred.item(), 0, edge_ms, 0, False
    else:
        # 反归一化 -> PIL -> PNG bytes
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_unnorm = img_tensor * std + mean
        img_unnorm = (img_unnorm * 255).clamp(0, 255).byte()
        pil_img = transforms.ToPILImage()(img_unnorm)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        payload = buf.getvalue()

        t2 = time.perf_counter()
        resp = requests.post(f'{CLOUD}/offload', data=payload)
        t3 = time.perf_counter()
        cloud_ms = (t3 - t2) * 1000

        return int(resp.text), len(payload), edge_ms, cloud_ms, True


if __name__ == '__main__':
    data = torch.load('data/cifar100.pt', weights_only=False)
    img = data['samples'][0]
    label = data['labels'][0]
    in_label = data['imagenet_labels'][0]
    print(f'CIFAR-10 标签: {label}, 映射 ImageNet: {in_label}')

    pred_a, comm_a, e_a, c_a = run_split(img)
    print(f'方案A 分割推理: pred={pred_a}, comm={comm_a}B, edge={e_a:.1f}ms, cloud={c_a:.1f}ms')

    pred_b, comm_b, e_b, c_b, up = run_offload(img)
    print(f'方案B 置信度卸载: pred={pred_b}, comm={comm_b}B, edge={e_b:.1f}ms, cloud={c_b:.1f}ms, uploaded={up}')
