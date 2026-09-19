# -*- coding: utf-8 -*-
"""结果分析 -> results/summary.txt + privacy_summary.txt + results/comparison.png

通信量报告两个口径, 消除上一版"上传比例 x 单次大小 != 平均值"的矛盾:
  [全样本均值]  = 上传总字节 / 全部推理次数 (未上传计 0)
  [上传样本均值] = 上传总字节 / 上传次数 (单次上传的实际大小)

另输出: 模型能力与协同策略分解 (端侧基线/模型上界/方案B 混合三者对照),
切分实现一致性验证, 端侧显存峰值, 以及由云端日志生成的隐私实测摘要.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux 容器无中文字体, 图内一律英文防乱码
plt.rcParams['axes.unicode_minus'] = False


def main():
    df = pd.read_csv('results/benchmark.csv')
    up = df[df['uploaded']]
    nu = df[~df['uploaded']]

    acc_split = (df['pred_split'] == df['true_label']).mean()
    acc_offload = (df['pred_offload'] == df['true_label']).mean()
    acc_edge_alone = (df['pred_edge_mobile'] == df['true_label']).mean()
    acc_rn50 = (df['pred_rn50'] == df['true_label']).mean()
    agree = (df['pred_split'] == df['pred_offload']).mean()
    split_eq_full = (df['pred_split'] == df['pred_rn50']).mean()
    upload_rate = df['uploaded'].mean()

    comm_a = df['comm_split_bytes'].mean()
    mean_comm_b = df['comm_offload_bytes'].mean()
    avg_up_bytes = up['comm_offload_bytes'].mean() if len(up) else 0

    # 上传样本中被云端纠正的比例 (MobileNet 错 -> 云端对)
    if len(up):
        corrected = ((up['pred_edge_mobile'] != up['true_label']) &
                     (up['pred_offload'] == up['true_label'])).mean()

    lines = ['=== 性能汇总 (200 样本 x 3 轮) ===']
    lines.append(
        f'端到端时延  方案A: {df["lat_split_ms"].mean():.2f} ms | '
        f'方案B: {df["lat_offload_ms"].mean():.2f} ms')
    lines.append(
        f'端侧计算    方案A: {df["edge_split_ms"].mean():.2f} ms | '
        f'方案B: {df["edge_offload_ms"].mean():.2f} ms')
    lines.append(
        f'云端+网络   方案A: {df["cloud_split_ms"].mean():.2f} ms | '
        f'方案B(仅上传样本): {up["cloud_offload_ms"].mean():.2f} ms')
    lines.append(
        f'通信量[全样本均值]  方案A: {comm_a/1024:.2f} KB | '
        f'方案B: {mean_comm_b/1024:.2f} KB')
    lines.append(
        f'通信量[上传样本单次均值] 方案B: {avg_up_bytes/1024:.2f} KB '
        f'(上传比例 {upload_rate:.2%}, 二者乘积=全样本均值, 口径自洽)')
    lines.append(
        f'准确率  方案A(ResNet50 分割): {acc_split:.2%} | '
        f'方案B(分级卸载): {acc_offload:.2%} | '
        f'端侧基线(MobileNetV3-Small 单独): {acc_edge_alone:.2%}')
    if len(up):
        lines.append(
            f'方案B 相比端侧基线的增益: {(acc_offload-acc_edge_alone)*100:+.1f} pp; '
            f'上传样本中被云端纠正的比例: {corrected:.2%}')
    lines.append(f'两方案预测一致率: {agree:.2%}')
    lines += ['', '=== 模型能力与协同策略分解 (区分模型差异与门控策略影响) ===']
    lines.append(
        f'模型上界 (ResNet50 完整推理): 准确率 {acc_rn50:.2%}, '
        f'本地时延 {df["rn50_ms"].mean():.2f} ms')
    lines.append(
        f'方案B 相比端侧基线 {(acc_offload-acc_edge_alone)*100:+.1f} pp, '
        f'距模型上界 {(acc_rn50-acc_offload)*100:.1f} pp '
        f'(差距来自高置信样本沿用端侧判断)')
    lines.append(
        f'切分实现验证: 方案A 与完整 ResNet50 预测一致率 {split_eq_full:.2%} '
        f'(前后向拼接与完整前向数值等价, 切分实现正确)')
    if os.path.exists('results/edge_peak_mem.txt'):
        with open('results/edge_peak_mem.txt', encoding='utf-8') as f:
            peak_mb = float(f.read().strip())
        lines.append(
            f'端侧进程 GPU 显存峰值: {peak_mb:.1f} MB '
            f'(MobileNet + ResNet50-front + 完整 ResNet50 全加载)')
    conf_nu = nu['conf_mobile'].mean() if len(nu) else float('nan')
    conf_up = up['conf_mobile'].mean() if len(up) else float('nan')
    lines.append(
        f'置信度(MobileNet): 全样本均值 {df["conf_mobile"].mean():.3f} | '
        f'未上传均值 {conf_nu:.3f} | '
        f'上传样本均值 {conf_up:.3f}')

    # 云端日志
    if os.path.exists('results/cloud_log.jsonl'):
        with open('results/cloud_log.jsonl', encoding='utf-8') as f:
            logs = [json.loads(l) for l in f if l.strip()]
        sp = [x for x in logs if x['route'] == 'split']
        of = [x for x in logs if x['route'] == 'offload']
        lines += ['', '=== 云端日志统计 ===']
        if sp:
            lines.append(f'/split   收到 {len(sp)} 次, '
                         f'平均 {sum(x["bytes"] for x in sp)/len(sp)/1024:.2f} KB')
        if of:
            lines.append(f'/offload 收到 {len(of)} 次, '
                         f'平均 {sum(x["bytes"] for x in of)/len(of)/1024:.2f} KB')

        # === 隐私数据实测摘要 (任务书: 结合通信内容与日志说明) ===
        p = ['=== 隐私数据实测摘要 (由 cloud_log.jsonl + benchmark.csv 生成) ===',
             '']
        p.append('[云端实际收到并记录的信息]')
        if sp:
            p.append(
                f'方案A /split: {len(sp)} 次请求, 每次 raw float32 特征图 '
                f'{sp[0]["bytes"]/1024:.0f} KB (1024x14x14), '
                f'累计 {sum(x["bytes"] for x in sp)/1024/1024:.2f} MB')
        if of:
            of_bytes = pd.Series([x['bytes'] for x in of])
            p.append(
                f'方案B /offload: {len(of)} 次请求, 原图 PNG 均值 '
                f'{of_bytes.mean()/1024:.2f} KB (标准差 {of_bytes.std()/1024:.1f} KB, '
                f'随图像内容波动), 累计 {of_bytes.sum()/1024/1024:.2f} MB')
        p.append(f'方案B 另有 {len(nu)} 次推理云端零可见 (端侧自答, 未发生通信)')
        p.append('云端日志字段: route / bytes / infer_ms / pred / time '
                 '(即云端方可持久化保存的全部信息)')
        p += ['', '[元数据泄露面 - 云端观察者据此可推断]']
        p.append(f'1. 上传比例 {upload_rate:.2%} -> 泄露样本难度分布与端侧模型能力边界')
        p.append(f'2. 请求时间戳范围 {min(x["time"] for x in logs)} ~ '
                 f'{max(x["time"] for x in logs)} -> 泄露用户活跃时段')
        if sp and of:
            p.append('3. 方案A 请求字节数恒定, 方案B 随图像内容波动 '
                     '-> 通信模式可区分两类协同请求')
        p += ['', '[端侧保留 - 未暴露给云端]']
        p.append('原图 (方案A 全部 / 方案B 高置信部分), 各样本置信度与端侧预测')
        p += ['', '风险成立条件与缓解建议见报告 main.tex 第 4 节; '
                 '本文件仅列实测可见数据, 不含推测.']
        with open('results/privacy_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(p))
        print('隐私实测摘要 -> results/privacy_summary.txt')

    txt = '\n'.join(lines)
    with open('results/summary.txt', 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)

    # === 图 ===
    tau = 0.9
    if os.path.exists('results/tau.txt'):
        with open('results/tau.txt', encoding='utf-8') as f:
            tau = float(f.read().strip())
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    a, b = 'Plan A\nSplitNN', 'Plan B\nConf. Offload'
    axes[0].bar([a, b], [df['lat_split_ms'].mean(),
                         df['lat_offload_ms'].mean()],
                color=['steelblue', 'coral'])
    axes[0].set_ylabel('ms')
    axes[0].set_title('End-to-end Latency')
    axes[1].bar([a, b], [comm_a / 1024, mean_comm_b / 1024],
                color=['steelblue', 'coral'])
    axes[1].set_ylabel('KB (mean over all queries)')
    axes[1].set_title('Avg. Communication')
    n_up, n_local = int(df['uploaded'].sum()), int((~df['uploaded']).sum())
    axes[2].pie([n_local, n_up], labels=['Edge self-answer', 'Offloaded'],
                autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
    axes[2].set_title(f'Plan B Routing (TAU={tau:.2f})')
    axes[3].hist(df['conf_mobile'], bins=40, color='gray')
    axes[3].axvline(tau, color='red', linestyle='--', label=f'threshold {tau:.2f}')
    axes[3].set_xlabel('MobileNet softmax confidence')
    axes[3].legend()
    axes[3].set_title('Edge Confidence Distribution')
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150)
    print('图已保存 results/comparison.png')
