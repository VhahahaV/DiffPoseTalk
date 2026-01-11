#!/bin/bash
# Style Encoder 一键训练脚本
# 使用方法: bash train_style_encoder.sh

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
SAVE_ITER=5000  # 每5000次迭代保存一次

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Style Encoder 训练脚本${NC}"
echo -e "${GREEN}============================================================${NC}"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: conda未安装或未在PATH中${NC}"
    exit 1
fi

# 激活conda环境
echo -e "${YELLOW}激活conda环境: diffposetalk_cu128${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffposetalk_cu128

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "diffposetalk_cu128" ]; then
    echo -e "${RED}错误: 无法激活conda环境 diffposetalk_cu128${NC}"
    echo -e "${YELLOW}请先创建环境: conda create -n diffposetalk_cu128 python=3.10${NC}"
    exit 1
fi

# 检查Python和PyTorch
echo -e "${YELLOW}检查环境...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo -e "${RED}错误: PyTorch未正确安装${NC}"
    exit 1
}

# 设置数据路径（根据baseline_prompt.md）
# 所有数据集都使用同一个根目录 /home/caizhuoqiang/Data
# 数据集路径通过 JSON 文件中的相对路径指定
DATA_ROOT="/home/caizhuoqiang/Data"
DATA_ROOTS=(
    "${DATA_ROOT}"
)

# JSON文件统一位置
JSON_ROOT="${JSON_ROOT:-${DATA_ROOT}/dataset_jsons/splits}"

# 训练JSON文件（使用绝对路径）
TRAIN_JSONS=(
    "${JSON_ROOT}/digital_human.json"
    "${JSON_ROOT}/MEAD_VHAP_train.json"
    "${JSON_ROOT}/MultiModal200_train.json"
)

# 验证JSON文件是否存在
echo -e "${YELLOW}检查数据文件...${NC}"
for json_path in "${TRAIN_JSONS[@]}"; do
    if [ ! -f "$json_path" ]; then
        echo -e "${YELLOW}警告: JSON文件不存在: $json_path${NC}"
    else
        echo -e "${GREEN}✓ 找到: $json_path${NC}"
    fi
done

# Style Encoder 训练参数
EXP_NAME="${EXP_NAME:-style_encoder_multi_dataset}"
STATS_FILE="${STATS_FILE:-stats_train.npz}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_MOTIONS="${N_MOTIONS:-100}"
FPS="${FPS:-25}"
MAX_ITER="${MAX_ITER:-100000}"
LR="${LR:-0.0001}"
TEMPERATURE="${TEMPERATURE:-0.1}"
VAL_ITER="${VAL_ITER:-10000}"
SAVE_ITER="${SAVE_ITER:-5000}"

# Style Encoder 模型参数
FEATURE_DIM="${FEATURE_DIM:-128}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-4}"
MLP_RATIO="${MLP_RATIO:-4}"

# 显示训练配置
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Style Encoder 训练配置${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "实验名称: $EXP_NAME"
echo "批次大小: $BATCH_SIZE"
echo "Motion数量: $N_MOTIONS"
echo "帧率: $FPS"
echo "最大迭代: $MAX_ITER"
echo "学习率: $LR"
echo "温度参数: $TEMPERATURE"
echo "验证频率: 每 $VAL_ITER 次迭代"
echo "保存频率: 每 $SAVE_ITER 次迭代"
echo "统计文件: $STATS_FILE"
echo "特征维度: $FEATURE_DIM"
echo "注意力头数: $N_HEADS"
echo "Transformer层数: $N_LAYERS"
echo "MLP比率: $MLP_RATIO"
echo -e "${GREEN}============================================================${NC}"

# 构建训练命令
TRAIN_CMD="python main_se.py \
    --mode train \
    --data_roots ${DATA_ROOTS[@]} \
    --data_jsons ${TRAIN_JSONS[@]} \
    --stats_file ${STATS_FILE} \
    --exp_name ${EXP_NAME} \
    --fps ${FPS} \
    --n_motions ${N_MOTIONS} \
    --batch_size ${BATCH_SIZE} \
    --max_iter ${MAX_ITER} \
    --lr ${LR} \
    --temperature ${TEMPERATURE} \
    --feature_dim ${FEATURE_DIM} \
    --n_heads ${N_HEADS} \
    --n_layers ${N_LAYERS} \
    --mlp_ratio ${MLP_RATIO} \
    --rot_repr aa \
    --no_head_pose \
    --val_iter ${VAL_ITER} \
    --save_iter ${SAVE_ITER}"

# 显示完整命令
echo -e "${YELLOW}执行训练命令:${NC}"
echo "$TRAIN_CMD"
echo ""

# 询问确认
read -p "是否开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    exit 0
fi

# 开始训练
echo -e "${GREEN}开始训练...${NC}"
echo -e "${GREEN}============================================================${NC}"

eval $TRAIN_CMD

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}Style Encoder 训练完成！${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${YELLOW}模型保存在: experiments/SE/${EXP_NAME}*/checkpoints/${NC}"
else
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}训练失败，退出码: $TRAIN_EXIT_CODE${NC}"
    echo -e "${RED}============================================================${NC}"
    exit $TRAIN_EXIT_CODE
fi

