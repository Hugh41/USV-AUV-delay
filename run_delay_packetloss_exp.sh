#!/bin/bash
# 运行丢包+延迟情况下的实验
# update_frequency=5, 分别测试 AUV=2, 3, 4
# 使用 Episode 575 的模型

echo "=========================================="
echo "开始运行丢包+延迟实验"
echo "配置:"
echo "  - Update Frequency: 5"
echo "  - 丢包模式: 开启"
echo "  - 延迟: 开启"
echo "  - 模型: Episode 575"
echo "=========================================="

# 检查并安装必要的依赖
echo ""
echo "检查Python依赖..."
python3 -c "import scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装 scipy..."
    python3 -m pip install scipy --user
fi

python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装 torch..."
    python3 -m pip install torch --user
fi

python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装 numpy..."
    python3 -m pip install numpy --user
fi

python3 -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装 matplotlib..."
    python3 -m pip install matplotlib --user
fi

echo "依赖检查完成"
echo ""

# 实验配置
UPDATE_FREQ=5
LOAD_EP=575
PACKET_LOSS_MODES="1"  # 1=开启丢包
REPEAT_NUM_2AUV=50     # 2AUV重复次数
REPEAT_NUM_3AUV=50     # 3AUV重复次数
REPEAT_NUM_4AUV=50     # 4AUV重复次数

# ==========================================
# 2AUV 实验（使用 models_ddpg_2）
# ==========================================
echo ""
echo "=========================================="
echo "运行2AUV实验（models_ddpg_2, Episode ${LOAD_EP}）"
echo "Update Frequency: ${UPDATE_FREQ}"
echo "丢包模式: 开启"
echo "重复次数: ${REPEAT_NUM_2AUV}"
echo "=========================================="

python3 compare_delay_stackelberg.py \
    --N_AUV 2 \
    --usv_update_frequency ${UPDATE_FREQ} \
    --load_ep ${LOAD_EP} \
    --repeat_num ${REPEAT_NUM_2AUV} \
    --packet_loss_modes "${PACKET_LOSS_MODES}" \
    --model_type td3

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 2AUV实验完成"
else
    echo ""
    echo "✗ 2AUV实验失败"
    exit 1
fi

# ==========================================
# 3AUV 实验（使用 models_ddpg_3）
# ==========================================
echo ""
echo "=========================================="
echo "运行3AUV实验（models_ddpg_3, Episode ${LOAD_EP}）"
echo "Update Frequency: ${UPDATE_FREQ}"
echo "丢包模式: 开启"
echo "重复次数: ${REPEAT_NUM_3AUV}"
echo "=========================================="

python3 compare_delay_stackelberg.py \
    --N_AUV 3 \
    --usv_update_frequency ${UPDATE_FREQ} \
    --load_ep ${LOAD_EP} \
    --repeat_num ${REPEAT_NUM_3AUV} \
    --packet_loss_modes "${PACKET_LOSS_MODES}" \
    --model_type td3

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 3AUV实验完成"
else
    echo ""
    echo "✗ 3AUV实验失败"
    exit 1
fi

# ==========================================
# 4AUV 实验（使用 models_ddpg_4）
# ==========================================
echo ""
echo "=========================================="
echo "运行4AUV实验（models_ddpg_4, Episode ${LOAD_EP}）"
echo "Update Frequency: ${UPDATE_FREQ}"
echo "丢包模式: 开启"
echo "重复次数: ${REPEAT_NUM_4AUV}"
echo "=========================================="

python3 compare_delay_stackelberg.py \
    --N_AUV 4 \
    --usv_update_frequency ${UPDATE_FREQ} \
    --load_ep ${LOAD_EP} \
    --repeat_num ${REPEAT_NUM_4AUV} \
    --packet_loss_modes "${PACKET_LOSS_MODES}" \
    --model_type td3

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 4AUV实验完成"
else
    echo ""
    echo "✗ 4AUV实验失败"
    exit 1
fi

# ==========================================
# 实验总结
# ==========================================
echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo ""
echo "实验结果保存在: delay_comparison_results/"
echo ""
echo "实验配置:"
echo "  - Update Frequency: ${UPDATE_FREQ}"
echo "  - 丢包模式: 开启"
echo "  - 延迟: 开启（固定延迟0.1s + 采样延迟最大0.333s）"
echo "  - 模型: Episode ${LOAD_EP}"
echo ""
echo "实验列表:"
echo "  ✓ 2AUV (重复${REPEAT_NUM_2AUV}次)"
echo "  ✓ 3AUV (重复${REPEAT_NUM_3AUV}次)"
echo "  ✓ 4AUV (重复${REPEAT_NUM_4AUV}次)"
echo ""

