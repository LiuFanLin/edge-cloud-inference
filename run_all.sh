#!/bin/bash
# 一键运行全部流程
set -e

echo "============================================"
echo "  端云协同推理性能对比 - 一键运行"
echo "============================================"

# 步骤1: 准备数据
echo ""
echo "[1/5] 准备测试数据..."
python prepare_data.py

# 步骤2: 启动云端服务
echo ""
echo "[2/5] 启动云端服务..."
python cloud_server.py &
CLOUD_PID=$!
echo "云端服务 PID: $CLOUD_PID"

# 等待云端启动
echo "等待云端服务启动..."
sleep 10

# 步骤3: 测试连接
echo ""
echo "[3/5] 测试端云连接..."
python edge_client.py

# 步骤4: 性能测试
echo ""
echo "[4/5] 运行性能测试 (100样本 x 3轮 x 2方案)..."
python run_benchmark.py

# 步骤5: 分析结果
echo ""
echo "[5/5] 分析结果 + 生成图表..."
python analyze.py

# 关闭云端
echo ""
echo "关闭云端服务..."
kill $CLOUD_PID

echo ""
echo "============================================"
echo "  全部完成! 结果在 results/ 目录"
echo "============================================"
echo "  benchmark.csv       - 原始数据"
echo "  comparison.png       - 对比图表"
echo "  summary.txt         - 性能汇总表"
echo "  privacy_analysis.md  - 隐私分析对比表"
echo "  cloud_log.json       - 云端日志"
echo "============================================"
