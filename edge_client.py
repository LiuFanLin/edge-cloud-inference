# -*- coding: utf-8 -*-
"""端侧客户端.

方案A 分割推理: ResNet50 切到 layer3 后, 端侧跑前半(conv1~layer3),
上传 1024x14x14 特征图, 云端跑后半(layer4 + fc). 每次推理必传.

方案B 置信度卸载 (端云模型分级, 参照 DDNN 两级架构):
端侧 MobileNetV3-Small (2.5M 参数) 自答; softmax 置信度低于阈值 TAU 时
上传原图, 由云端 ResNet50 (25.6M 参数) 重判. 端轻云强, 门控才有增益.
"""
import io
import time

import requests
import torch
from torchvision import transforms

from models import get_resnet50, get_mobilenet_v3_small, split_resnet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUT = 'layer3'
C, H, W = 1024, 14, 14   # layer3 后特征图形状
TAU = 0.9                # 方案B 阈值, 由 run_benchmark 用校准集分位数确定
CLOUD = 'http://localhost:5000'

print(f'加载端侧模型 (device={DEVICE})...')
_resnet = get_resnet50().to(DEVICE).eval()
_front, _, _ = split_resnet(_resnet, CUT)
_mobile = get_mobilenet_v3_small().to(DEVICE).eval()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


def mobile_conf(img_tensor):
    """端侧小模型预测, 返回 (置信度, 预测)."""
    with torch.no_grad():
        prob = torch.softmax(
            _mobile(img_tensor.unsqueeze(0).to(DEVICE)), dim=1)
        conf, pred = prob.max(dim=1)
    return conf.item(), pred.item()


def run_split(img_tensor):
    """方案A. 返回 (pred, comm_bytes, edge_ms, cloud_ms).

    edge_ms 含 GPU 同步时间: CUDA 前向是异步的, 必须 synchronize 后
    再取时刻, 否则 GPU 上时延被严重低估.
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        feat = _front(img_tensor.unsqueeze(0).to(DEVICE))
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    payload = feat.cpu().numpy().tobytes()
    t1 = time.perf_counter()
    r = requests.post(f'{CLOUD}/split', data=payload,
                      params={'c': C, 'h': H, 'w': W}, timeout=60)
    t2 = time.perf_counter()
    return int(r.text), len(payload), (t1 - t0) * 1000, (t2 - t1) * 1000


def run_offload(img_tensor):
    """方案B. 返回 (pred_final, comm_bytes, edge_ms, cloud_ms,
    conf, uploaded, pred_edge_mobile).

    edge_ms = MobileNet 推理 + PNG 编码 (均为端侧处理);
    cloud_ms = 上传 + 云端重判 + 返回. 端到端 = edge_ms + cloud_ms,
    编码耗时不遗漏.
    """
    t0 = time.perf_counter()
    conf, pred_edge = mobile_conf(img_tensor)  # 内部 .item() 已隐式同步
    if conf >= TAU:
        t1 = time.perf_counter()
        return pred_edge, 0, (t1 - t0) * 1000, 0.0, conf, False, pred_edge
    # 低置信: 反归一化 -> PNG 编码 -> 上传云端重判
    x = img_tensor.to(DEVICE) * STD + MEAN
    x = (x * 255).clamp(0, 255).byte().cpu()
    pil = transforms.ToPILImage()(x)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    payload = buf.getvalue()
    t2 = time.perf_counter()  # 此刻端侧处理(推理+编码)全部完成
    r = requests.post(f'{CLOUD}/offload', data=payload, timeout=60)
    t3 = time.perf_counter()
    return (int(r.text), len(payload), (t2 - t0) * 1000,
            (t3 - t2) * 1000, conf, True, pred_edge)


if __name__ == '__main__':
    data = torch.load('data/test250.pt', weights_only=False)
    img, label = data['samples'][0], data['labels'][0]
    print(f'真实标签 (ImageNet index): {label}')
    pa = run_split(img)
    print(f'方案A: pred={pa[0]}, comm={pa[1]}B, '
          f'edge={pa[2]:.1f}ms, cloud={pa[3]:.1f}ms')
    pb = run_offload(img)
    print(f'方案B: pred={pb[0]}, comm={pb[1]}B, conf={pb[4]:.3f}, '
          f'uploaded={pb[5]}')
