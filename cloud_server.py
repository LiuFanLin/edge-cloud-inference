# -*- coding: utf-8 -*-
"""云端服务 (Flask):
/split   方案A: 接收中间特征图, 跑 ResNet50 后半 (layer4 + fc)
/offload 方案B: 接收原图 PNG, 跑完整 ResNet50 (端云模型分级, 云侧为强模型)
每请求记录 jsonl 日志 (路由/字节数/推理耗时/预测).
"""
import io
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from flask import Flask, request
from PIL import Image
from torchvision import transforms

from models import get_resnet50, split_resnet

app = Flask(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CUT = 'layer3'  # 主切分点: layer3 之后, 特征 1024x14x14

os.makedirs('results', exist_ok=True)
LOG = 'results/cloud_log.jsonl'

print(f'加载云端模型 (device={DEVICE}, 切分点={CUT})...')
_resnet = get_resnet50().to(DEVICE).eval()
_front, _back, _fc = split_resnet(_resnet, CUT)

# 上传的 PNG 已是端侧 CenterCrop 224 的结果, 云端只做 ToTensor+Normalize,
# 不得再 Resize/Crop —— 否则等效于对输入二次裁剪, 改变样本内容.
TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def log(entry: dict):
    entry['time'] = datetime.now().isoformat(timespec='milliseconds')
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


@app.route('/split', methods=['POST'])
def split_infer():
    """方案A: 接收中间特征图, shape 由 query 参数 c/h/w 传入."""
    t0 = time.perf_counter()
    data = request.get_data()
    c = int(request.args.get('c', '1024'))
    h = int(request.args.get('h', '14'))
    w = int(request.args.get('w', '14'))
    feat = np.frombuffer(data, dtype=np.float32).reshape(1, c, h, w).copy()
    with torch.no_grad():
        x = torch.from_numpy(feat).to(DEVICE)
        x = _back(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        out = _fc(x)
    pred = int(out.argmax(1).item())
    infer_ms = (time.perf_counter() - t0) * 1000
    log({'route': 'split', 'bytes': len(data),
         'infer_ms': round(infer_ms, 2), 'pred': pred})
    return str(pred)


@app.route('/offload', methods=['POST'])
def offload_infer():
    """方案B: 接收原图 PNG, 用完整 ResNet50 重判."""
    t0 = time.perf_counter()
    data = request.get_data()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    tensor = TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _resnet(tensor)
    pred = int(out.argmax(1).item())
    infer_ms = (time.perf_counter() - t0) * 1000
    log({'route': 'offload', 'bytes': len(data),
         'infer_ms': round(infer_ms, 2), 'pred': pred})
    return str(pred)


if __name__ == '__main__':
    print('云端服务启动 http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000)
