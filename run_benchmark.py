# -*- coding: utf-8 -*-
"""性能测试:
1) 前 50 张做方案B 阈值校准 (取 MobileNet 置信度 30% 分位数, 截断 [0.5,0.99]),
   使上传比例可控且可复现;
2) 后 200 张正式测试, 3 轮, 两方案, 记录时延/通信/置信度/预测.
输出 results/benchmark.csv
"""
import csv
import io
import os
import time

import requests
import torch
from PIL import Image as PILImage

import edge_client
from edge_client import run_split, run_offload

os.makedirs('results', exist_ok=True)
CALIB, N, ROUNDS, WARMUP = 50, 200, 3, 5


def main():
    data = torch.load('data/test250.pt', weights_only=False)
    samples, labels = data['samples'], data['labels']
    assert len(samples) >= CALIB + N, '样本不足, 请重跑 prepare_data.py'

    # 显存峰值统计覆盖全流程 (校准 + 基线 + 双方案)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- 阈值校准 ---
    confs = []
    for i in range(CALIB):
        c, _ = edge_client.mobile_conf(samples[i])
        confs.append(c)
    confs.sort()
    tau = confs[int(0.30 * len(confs))]  # 30% 分位 -> 目标上传比例约 30%
    tau = min(max(tau, 0.50), 0.99)
    edge_client.TAU = tau
    with open('results/tau.txt', 'w', encoding='utf-8') as f:
        f.write(f'{tau:.4f}')
    print(f'校准完成: 阈值 TAU = {tau:.4f} (目标上传比例约 30%)')

    # --- 预热 (校准集样本, 不计入正式数据) ---
    print('预热...')
    for i in range(WARMUP):
        run_split(samples[i])
        run_offload(samples[i])
    # 校准样本可能全部高置信而未触发上传, 显式预热一次云端 /offload 路由,
    # 避免首个真实上传样本承担模型 warm-up 开销.
    _buf = io.BytesIO()
    PILImage.new('RGB', (224, 224)).save(_buf, format='PNG')
    requests.post(f'{edge_client.CLOUD}/offload',
                  data=_buf.getvalue(), timeout=60)
    # 预热请求也写入了云端日志, 清空以保证正式统计只含测试数据
    if os.path.exists('results/cloud_log.jsonl'):
        os.remove('results/cloud_log.jsonl')

    # --- 纯云模型能力基线: 完整 ResNet50 对全部正式样本各推理一次 ---
    # 涉及不同模型时, 区分模型本身的能力差异与协同策略的影响:
    # 端侧基线(MobileNetV3) / 模型上界(ResNet50) / 方案B 混合, 三者对照分解.
    # 同时用 pred_rn50 验证方案A 切分拼接与完整前向的数值等价性.
    print('完整 ResNet50 基线推理...')
    rn50_preds, rn50_times = [], []
    with torch.no_grad():
        for j in range(N):
            x = samples[CALIB + j].unsqueeze(0).to(edge_client.DEVICE)
            t0 = time.perf_counter()
            out = edge_client._resnet(x)
            if edge_client.DEVICE == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            rn50_preds.append(int(out.argmax(dim=1).item()))
            rn50_times.append((t1 - t0) * 1000)
    print(f'基线完成, 平均时延 {sum(rn50_times)/N:.2f} ms')

    rows = []
    for r in range(ROUNDS):
        print(f'=== 第 {r + 1}/{ROUNDS} 轮 ===')
        for j in range(N):
            i = CALIB + j
            img, y = samples[i], labels[i]
            ps, cs, es, vs = run_split(img)
            pf, co, eo, vo, conf, up, pe = run_offload(img)
            rows.append({
                'round': r, 'sample_id': i, 'true_label': y,
                'pred_edge_mobile': pe, 'conf_mobile': round(conf, 4),
                'uploaded': up,
                'pred_split': ps, 'lat_split_ms': round(es + vs, 2),
                'edge_split_ms': round(es, 2),
                'cloud_split_ms': round(vs, 2),
                'comm_split_bytes': cs,
                'pred_offload': pf, 'lat_offload_ms': round(eo + vo, 2),
                'edge_offload_ms': round(eo, 2),
                'cloud_offload_ms': round(vo, 2),
                'comm_offload_bytes': co,
                'pred_rn50': rn50_preds[j],
                'rn50_ms': round(rn50_times[j], 2),
            })
            if (j + 1) % 50 == 0:
                print(f'  {j + 1}/{N}')

    with open('results/benchmark.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    # 端侧进程显存峰值 (MobileNet + ResNet50-front + 完整 ResNet50 全加载时的最坏情况)
    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    with open('results/edge_peak_mem.txt', 'w', encoding='utf-8') as f:
        f.write(f'{peak_mb:.1f}')
    print(f'完成 {len(rows)} 条 -> results/benchmark.csv '
          f'(端侧 GPU 显存峰值 {peak_mb:.1f} MB)')


if __name__ == '__main__':
    main()
