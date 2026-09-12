# 端云协同推理性能对比与隐私安全分析

## 快速开始

```bash
# 1. 安装依赖
bash setup.sh

# 2. 一键运行（自动启动云端+测试+分析）
bash run_all.sh

# 或者分步运行：
python prepare_data.py        # 准备数据
python cloud_server.py &      # 启动云端（后台）
sleep 10                      # 等待启动
python edge_client.py         # 测试连接
python run_benchmark.py       # 性能测试
python analyze.py             # 分析画图
```

## 结果文件

| 文件 | 内容 |
|------|------|
| `results/benchmark.csv` | 每次推理的原始数据 |
| `results/comparison.png` | 4 张对比图 |
| `results/summary.txt` | 性能汇总表 |
| `results/privacy_analysis.md` | 隐私安全对比表 |
| `results/cloud_log.json` | 云端接收到的所有请求日志 |

## 两种协同方案

### 方案A: 分割推理 (SplitNN)
- ResNet18 切到 layer2 之后
- 端侧跑前半 → 输出 128×28×28 特征图 → 上传
- 云端跑后半 → 返回预测
- **每次都传特征图（约 401KB）**

### 方案B: 置信度驱动卸载
- 端侧跑完整 ResNet18 + softmax
- 置信度 ≥ 0.8 → 直接返回端侧结果（不上传）
- 置信度 < 0.8 → 上传原图给云端重判
- **仅低置信度样本传原图**

## 测试条件

- 数据集: CIFAR-10 测试集 100 张，resize 到 224×224
- 模型: ResNet18 (ImageNet 预训练)
- 部署: 同一台机器，两个进程模拟端云
- 通信: HTTP (localhost:5000)
- 重复: 3 轮 × 100 样本 = 300 次/方案
- 计时: 端到端 = 数据预处理后 → 收到预测结果
