# -*- coding: utf-8 -*-
"""分析结果 + 画图 + 生成隐私对比表"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('results/benchmark.csv')

# === 准确率 ===
# 以 ImageNet 映射后的真实标签为准
acc_a = (df['pred_split'] == df['true_imagenet']).mean()
acc_b = (df['pred_offload'] == df['true_imagenet']).mean()
consistency = (df['pred_split'] == df['pred_offload']).mean()

# === 方案B 上传比例 ===
upload_rate = df['uploaded'].mean()
upload_count = df['uploaded'].sum()

# === 平均指标 ===
lat_a = df['latency_split_ms'].mean()
lat_b = df['latency_offload_ms'].mean()
comm_a = df['comm_split_bytes'].mean()
comm_b = df['comm_offload_bytes'].mean()
edge_a = df['edge_split_ms'].mean()
edge_b = df['edge_offload_ms'].mean()
cloud_a = df['cloud_split_ms'].mean()
cloud_b = df['cloud_offload_ms'].mean()

# === 画图 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 端到端时延对比
axes[0, 0].bar(['SplitNN\n(A)', 'Offload\n(B)'],
               [lat_a, lat_b], color=['steelblue', 'coral'])
axes[0, 0].set_ylabel('Latency (ms)')
axes[0, 0].set_title('End-to-End Latency')

# 图2: 通信量对比
axes[0, 1].bar(['SplitNN\n(A)', 'Offload\n(B)'],
               [comm_a / 1024, comm_b / 1024], color=['steelblue', 'coral'])
axes[0, 1].set_ylabel('Avg Comm (KB)')
axes[0, 1].set_title('Average Communication')

# 图3: 端侧 vs 云端耗时
x = np.arange(2)
w = 0.35
axes[1, 0].bar(x - w/2, [edge_a, edge_b], w, label='Edge', color='steelblue')
axes[1, 0].bar(x + w/2, [cloud_a, cloud_b], w, label='Cloud+Network', color='coral')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['SplitNN (A)', 'Offload (B)'])
axes[1, 0].set_ylabel('Time (ms)')
axes[1, 0].set_title('Edge vs Cloud Time')
axes[1, 0].legend()

# 图4: 方案B 端云分流饼图
total_count = len(df)
edge_count = total_count - upload_count
axes[1, 1].pie([edge_count, upload_count],
               labels=[f'Edge-only ({edge_count})', f'Uploaded ({upload_count})'],
               autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
axes[1, 1].set_title('Offload (B): Edge vs Cloud Split')

plt.tight_layout()
plt.savefig('results/comparison.png', dpi=150)
print('图表已保存到 results/comparison.png')

# === 汇总表 ===
summary = f"""============================================
   性能对比汇总表
============================================

测试条件:
  样本数: 100 (CIFAR-10 test set, resized to 224x224)
  重复轮数: 3 (共 300 次推理/方案)
  模型: ResNet18 (ImageNet 预训练)
  部署: 同一台机器, 两个进程模拟端云
  通信: HTTP (localhost:5000)
  计时口径: 端到端 = 数据预处理后 -> 收到预测结果
  预热: 5 张 (不计入统计)

--------------------------------------------
指标                  方案A(SplitNN)    方案B(Offload)
--------------------------------------------
端到端时延 (ms)        {lat_a:<16.2f} {lat_b:<16.2f}
端侧计算 (ms)          {edge_a:<16.2f} {edge_b:<16.2f}
云端+网络 (ms)         {cloud_a:<16.2f} {cloud_b:<16.2f}
平均通信量 (bytes)     {comm_a:<16.0f} {comm_b:<16.0f}
平均通信量 (KB)        {comm_a/1024:<16.2f} {comm_b/1024:<16.2f}
准确率 (ImageNet映射)   {acc_a:<16.2%} {acc_b:<16.2%}
上传比例               {'100%':<16} {upload_rate:<16.2%}
两方案预测一致率        {consistency:<16.2%}
--------------------------------------------
"""
print(summary)

with open('results/summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

# === 隐私分析表 ===
privacy = """============================================
   隐私安全对比表
============================================

方案A: 分割推理 (SplitNN)
  协同机制: ResNet18 按层切分, 端侧跑前半(conv1~layer2),
           输出 128x28x28 中间特征图上传, 云端跑后半(layer3~fc)
  传输与保存的信息:
    - 端侧 -> 云端: 128x28x28 float32 特征图 (约 401KB/次)
    - 云端可记录: 特征图内容 + 请求时间 + 请求来源
    - 端侧保留: 原始图像 (不上传)
  可能风险:
    1. 特征反演重建: 中间特征图含丰富语义信息,
       攻击者拿到特征图 + 模型后半部分可能重建原图
       (文献依据: "Deep Leakage from Gradients" 等研究表明
       中间特征可被用于重建输入)
    2. 语义信息泄露: 128x28x28 特征图大小接近原图,
       信息密度高, 含纹理/形状/对象信息
    3. 通信链路窃听: 传输 401KB 数据, 中间人可截获
  风险成立条件:
    - 攻击者获取特征图 + 模型后半部分权重
    - 或: 通信链路未加密, 中间人可截获
    - 或: 云端服务方不可信, 持久化存储特征图
  缓解建议:
    - 特征加噪: 在特征图上添加少量噪声, 降低反演质量
      (影响: 可能降低预测准确率 1-3%)
    - 特征降维: 用 PCA 将 128x28x28 降到更低维度
      (影响: 增加端侧计算, 减少通信量)
    - 链路加密: 使用 HTTPS 替代 HTTP
      (影响: 增加少量加密开销)
    - 云端不持久化: 处理完即删除特征图
      (影响: 无性能影响, 但失去日志能力)

--------------------------------------------

方案B: 置信度驱动卸载 (Confidence-based Offload)
  协同机制: 端侧跑完整 ResNet18, 取 softmax 最大值作为置信度,
           置信度 >= 0.8 直接返回端侧结果;
           置信度 < 0.8 上传原图给云端重判
  传输与保存的信息:
    - 高置信度样本: 不传输任何数据
    - 低置信度样本: 原始图像 PNG (约 100-150KB/次)
    - 云端可记录: 低置信度样本原图 + 请求时间
    - 端侧保留: 所有样本的预测结果 + 置信度值
  可能风险:
    1. 原始图像泄露: 低置信度样本原图直接上传,
       云端获取完整原图 (最敏感信息)
    2. 元数据泄露: 云端知道哪些样本端侧不确定,
       这本身泄露了样本难度分布信息
       (实际观察: 云端日志可统计上传比例)
    3. 置信度阈值泄露模型特性: 攻击者可通过
       观察上传模式推断端侧模型的能力边界
  风险成立条件:
    - 云端不可信 + 低置信度样本被记录
    - 或: 通信链路未加密, 中间人截获原图
    - 或: 攻击者可通过多次请求推断阈值
  缓解建议:
    - 提高阈值: 阈值从 0.8 提到 0.95,
       减少上传比例
      (影响: 更多样本用端侧结果, 可能降低整体准确率)
    - 端侧模型增强: 用更大模型提升基础置信度
      (影响: 增加端侧计算开销)
    - 原图加密传输: 使用 HTTPS
      (影响: 增加少量加密开销)
    - 差分隐私: 在上传前对原图加噪
      (影响: 可能降低云端预测准确率)

--------------------------------------------

综合对比:
  通信量: 方案A 每次传 401KB, 方案B 仅低置信度样本传 100-150KB
  原图暴露: 方案A 不传原图, 方案B 部分样本传原图
  特征泄露: 方案A 传中间特征(可反演), 方案B 不传特征
  元数据泄露: 方案A 无(每次都一样), 方案B 有(上传模式泄露信息)
  
  关键结论:
    "未上传原图" 不等于 "没有隐私风险"
    - 方案A 不传原图但传大特征图, 反演攻击有文献支持
    - 方案B 有时不传任何数据, 但上传条件本身泄露元信息
"""

print(privacy)
with open('results/privacy_analysis.md', 'w', encoding='utf-8') as f:
    f.write(privacy)

# === 云端日志分析 ===
if os.path.exists('results/cloud_log.json'):
    with open('results/cloud_log.json', 'r', encoding='utf-8') as f:
        clog = json.load(f)
    split_calls = [x for x in clog if x['endpoint'] == '/split']
    offload_calls = [x for x in clog if x['endpoint'] == '/offload']
    log_summary = f"""
============================================
   云端日志分析 (隐私分析辅助)
============================================
方案A /split 调用次数: {len(split_calls)}
  每次接收数据: {split_calls[0]['bytes'] if split_calls else 0} bytes
  数据类型: 中间特征图 (128x28x28 float32)
  
方案B /offload 调用次数: {len(offload_calls)}
  每次接收数据: {offload_calls[0]['bytes'] if offload_calls else 0} bytes (平均)
  数据类型: 原始图像 PNG
  实际上传比例: {len(offload_calls) / (len(split_calls) + len(offload_calls)) * 100:.1f}%

云端可见信息:
  方案A: 每次请求的中间特征图 (可重建)
  方案B: 仅低置信度样本的原图 (最敏感)
"""
    print(log_summary)
    with open('results/cloud_log_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(log_summary)

print('\n所有结果已保存到 results/ 目录:')
print('  benchmark.csv       - 原始数据')
print('  comparison.png       - 对比图表')
print('  summary.txt         - 性能汇总表')
print('  privacy_analysis.md  - 隐私分析对比表')
print('  cloud_log.json       - 云端日志')
print('  cloud_log_analysis.txt - 云端日志分析')
