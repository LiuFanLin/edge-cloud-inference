#!/bin/bash
set -e
echo "[1/6] 准备数据 (imagenette2-320, 250 张)..."
python prepare_data.py
echo "[2/6] 模型规模评估 (参数量/FLOPs/切分点分解)..."
python models.py
mkdir -p results
rm -f results/cloud_log.jsonl
echo "[3/6] 启动云端服务 (首次运行需下载 ResNet50 权重约 100MB)..."
python cloud_server.py &
CLOUD_PID=$!
trap 'kill $CLOUD_PID 2>/dev/null' EXIT
echo "云端 PID: $CLOUD_PID, 轮询等待就绪 (最多 6 分钟)..."
for i in $(seq 1 180); do
    if curl -s -o /dev/null http://localhost:5000/; then
        echo "云端已就绪"
        break
    fi
    if ! kill -0 $CLOUD_PID 2>/dev/null; then
        echo "错误: 云端进程异常退出"
        exit 1
    fi
    if [ "$i" = "180" ]; then
        echo "错误: 等待云端超时"
        exit 1
    fi
    sleep 2
done
echo "[4/6] 连接测试 (首次运行会再下载端侧模型权重)..."
python edge_client.py
echo "[5/6] 性能测试 (阈值校准 + 200 样本 x 3 轮)..."
python run_benchmark.py
kill $CLOUD_PID 2>/dev/null || true
echo "[6/6] 分析结果..."
python analyze.py
echo "全部完成: results/summary.txt / comparison.png / model_profile.txt / benchmark.csv"
