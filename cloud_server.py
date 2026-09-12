# -*- coding: utf-8 -*-
"""云端服务：Flask 提供 split 和 offload 两个接口"""
import numpy as np
import torch
import io
from flask import Flask, request
from models import get_resnet18, split_resnet18
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

print('加载云端模型...')
resnet = get_resnet18()
front, back, fc = split_resnet18(resnet)
resnet.eval()
back.eval()
fc.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 记录云端接收到的数据（用于隐私分析）
cloud_log = []


@app.route('/split', methods=['POST'])
def split_infer():
    """方案A: 接收中间特征图 (1x128x28x28 float32)"""
    data = request.get_data()
    cloud_log.append({
        'endpoint': '/split',
        'bytes': len(data),
        'content_type': 'feature_map_128x28x28',
    })
    feat = np.frombuffer(data, dtype=np.float32).reshape(1, 128, 28, 28)
    with torch.no_grad():
        x = torch.from_numpy(feat.copy())
        x = back(x)                                  # 1x512x7x7
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # 1x512x1x1
        x = torch.flatten(x, 1)                      # 1x512
        out = fc(x)
    pred = out.argmax(1).item()
    return str(pred)


@app.route('/offload', methods=['POST'])
def offload_infer():
    """方案B: 接收原图 PNG bytes"""
    img_bytes = request.get_data()
    cloud_log.append({
        'endpoint': '/offload',
        'bytes': len(img_bytes),
        'content_type': 'raw_image_png',
    })
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = resnet(tensor)
    pred = out.argmax(1).item()
    return str(pred)


@app.route('/log', methods=['GET'])
def get_log():
    """返回云端日志（用于隐私分析）"""
    import json
    return json.dumps(cloud_log, ensure_ascii=False)


if __name__ == '__main__':
    print('云端服务启动 http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000)
