# -*- coding: utf-8 -*-
"""性能测试主脚本：100 样本 x 3 轮 x 2 方案"""
import torch
import time
import csv
import os
import requests
from edge_client import run_split, run_offload

os.makedirs('results', exist_ok=True)

data = torch.load('data/cifar100.pt', weights_only=False)
samples = data['samples']
labels = data['labels']
imagenet_labels = data['imagenet_labels']

# 预热 5 张
print('预热...')
for i in range(5):
    run_split(samples[i])
    run_offload(samples[i])

results = []
NUM_ROUNDS = 3

for round_idx in range(NUM_ROUNDS):
    print(f'\n=== 第 {round_idx + 1} 轮 ===')
    for i in range(100):
        img = samples[i]
        true_in = imagenet_labels[i]
        true_cifar = labels[i]

        # 方案A
        t0 = time.perf_counter()
        pred_a, comm_a, e_a, c_a = run_split(img)
        t1 = time.perf_counter()
        latency_a = (t1 - t0) * 1000

        # 方案B
        t0 = time.perf_counter()
        pred_b, comm_b, e_b, c_b, uploaded = run_offload(img)
        t1 = time.perf_counter()
        latency_b = (t1 - t0) * 1000

        results.append({
            'round': round_idx,
            'sample_id': i,
            'true_cifar': true_cifar,
            'true_imagenet': true_in,
            'pred_split': pred_a,
            'latency_split_ms': round(latency_a, 2),
            'edge_split_ms': round(e_a, 2),
            'cloud_split_ms': round(c_a, 2),
            'comm_split_bytes': comm_a,
            'pred_offload': pred_b,
            'latency_offload_ms': round(latency_b, 2),
            'edge_offload_ms': round(e_b, 2),
            'cloud_offload_ms': round(c_b, 2),
            'comm_offload_bytes': comm_b,
            'uploaded': uploaded,
        })

        if (i + 1) % 20 == 0:
            print(f'  已完成 {i + 1}/100')

# 获取云端日志
try:
    cloud_log = requests.get('http://localhost:5000/log', timeout=5).json()
    import json
    with open('results/cloud_log.json', 'w', encoding='utf-8') as f:
        json.dump(cloud_log, f, ensure_ascii=False, indent=2)
    print(f'云端日志已保存 ({len(cloud_log)} 条记录)')
except Exception as e:
    print(f'获取云端日志失败: {e}')

# 保存 CSV
with open('results/benchmark.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f'\n结果已保存到 results/benchmark.csv ({len(results)} 条记录)')
