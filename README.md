# 端云协同推理性能对比与隐私安全分析


## 快速开始

```bash
# 1. 安装依赖（若镜像已自带 torch/torchvision，只需: pip install flask requests pandas matplotlib）
bash setup.sh

# 2. 一键运行（数据准备 -> 模型评估 -> 启动云端 -> 阈值校准 -> 600 次推理 -> 分析出图）
bash run_all.sh
```

分步运行：

```bash
python prepare_data.py     # 下载 imagenette 并生成 data/test250.pt
python models.py           # 模型参数量/FLOPs/三个切分点分解
python cloud_server.py &   # 启动云端服务（首次运行需下载权重，等待加载完成）
python edge_client.py      # 连接测试
python run_benchmark.py    # 阈值校准 + 正式测试 -> results/benchmark.csv
python analyze.py          # 汇总分析 -> summary.txt + comparison.png
```

## 两种协同方案

### 方案A: 分割推理 (SplitNN)
- ResNet50 切到 layer3 之后（切分点对比见 `results/model_profile.txt`）
- 端侧 conv1~layer3 输出 1024×14×14 特征图上传，云端 layer4+fc 返回预测
- **每次必传 raw float32 特征（约 784KB，未压缩）**

### 方案B: 置信度驱动卸载（端云模型分级，参照 DDNN 两级架构）
- 端侧 MobileNetV3-Small (2.5M 参数) 先行推理
- softmax 置信度 ≥ τ → 端侧自答，不上传
- 置信度 < τ → 上传原图 PNG，云端 ResNet50 (25.6M 参数) 重判
- **τ 不人为拍定**：由 50 张校准样本置信度的 30% 分位数自动校准（写入 `results/tau.txt`）

## 测试条件

- 数据集: imagenette2-320 验证集（ImageNet 官方 10 类子集，与预训练分布一致），每类 25 张共 250 张；前 50 张为校准集，后 200 张为正式测试集
- 模型: torchvision ImageNet 预训练 ResNet50 / MobileNetV3-Small
- 部署: 同一台机器两个进程模拟端云，HTTP localhost:5000
- 重复: 3 轮 × 200 样本 = 600 次/方案，预热 5 张不计入
- 计时: 端到端 = 端侧前向开始 → 收到最终预测；GPU 前向后 synchronize，排除 CUDA 异步计时误差
- 通信量两口径分开报告: 全样本均值（未上传计 0）/ 上传样本单次均值，二者满足 全样本均值 = 上传比例 × 单次均值

## 结果文件

| 文件 | 内容 |
|------|------|
| `results/benchmark.csv` | 每次 inference 的原始记录（预测/时延分解/通信字节/置信度） |
| `results/summary.txt` | 性能汇总（两口径通信量、准确率、门控增益、纠正比例） |
| `results/comparison.png` | 时延/通信量/分流比例/置信度分布 4 图 |
| `results/tau.txt` | 校准得到的阈值 τ |
| `results/model_profile.txt` | 模型规模与三个候选切分点分解 |
| `results/cloud_log.jsonl` | 云端每请求日志（路由/字节数/推理耗时/预测/时间戳） |
| `results/privacy_summary.txt` | 隐私实测摘要：云端可见信息/元数据泄露面/端侧保留（由日志自动生成） |
| `results/edge_peak_mem.txt` | 端侧进程 GPU 显存峰值 |

