#!/bin/bash
# 一键推理脚本
# 使用方法: bash infer.sh [--exp_name EXP_NAME] [--iter ITER]

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}DiffPoseTalk 推理脚本${NC}"
echo -e "${GREEN}============================================================${NC}"

# 解析命令行参数
EXP_NAME=""
ITER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --iter)
            ITER="$2"
            shift 2
            ;;
        *)
            echo -e "${YELLOW}未知参数: $1${NC}"
            shift
            ;;
    esac
done

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

# 测试JSON文件（相对于data_root）
TEST_JSONS=(
    # "dataset_jsons/splits/MEAD_VHAP_test.json"
    "dataset_jsons/splits/MultiModal200_test.json"
)

# 验证JSON文件是否存在（使用公共 DATA_ROOT）
echo -e "${YELLOW}检查数据文件...${NC}"
for json_rel in "${TEST_JSONS[@]}"; do
    json_path="${DATA_ROOT}/${json_rel}"
    if [ ! -f "$json_path" ]; then
        echo -e "${YELLOW}警告: JSON文件不存在: $json_path${NC}"
    else
        echo -e "${GREEN}✓ 找到: $json_path${NC}"
    fi
done

# 推理参数（从环境变量或默认值）
EXP_NAME="${EXP_NAME:-multi_dataset_train-260105_164054}"
ITER="${ITER:-40000}"
STATS_FILE="${STATS_FILE:-stats_train.npz}"
STYLE_ENC_CKPT="${STYLE_ENC_CKPT:-/home/caizhuoqiang/Code/audio_driven_baseline/DiffPoseTalk/experiments/SE/style_encoder_multi_dataset-260104_221342/checkpoints/iter_0100000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/metrics}"
N_REPETITIONS="${N_REPETITIONS:-1}"

# 检查模型文件是否存在
MODEL_DIR="experiments/DPT/${EXP_NAME}"
MODEL_FILE="${MODEL_DIR}/checkpoints/iter_$(printf "%07d" ${ITER}).pt"

if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}错误: 模型文件不存在: $MODEL_FILE${NC}"
    echo -e "${YELLOW}请检查实验名称和迭代次数${NC}"
    echo -e "${YELLOW}可用模型:${NC}"
    find experiments/DPT -name "iter_*.pt" -type f 2>/dev/null | head -5 || echo "未找到模型文件"
    exit 1
fi

echo -e "${GREEN}✓ 找到模型文件: $MODEL_FILE${NC}"

# 显示推理配置
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}推理配置${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "实验名称: $EXP_NAME"
echo "迭代次数: $ITER"
echo "输出目录: $OUTPUT_DIR"
echo "重复次数: $N_REPETITIONS"
echo "统计文件: $STATS_FILE"
echo "Style Encoder: $STYLE_ENC_CKPT"
echo -e "${GREEN}============================================================${NC}"

# 检查style encoder文件
if [ ! -f "$STYLE_ENC_CKPT" ]; then
    echo -e "${YELLOW}警告: Style Encoder 文件不存在: $STYLE_ENC_CKPT${NC}"
    echo -e "${YELLOW}将尝试使用模型中保存的路径或关闭style guiding${NC}"
fi

# 构建推理命令
INFER_CMD="CUDA_VISIBLE_DEVICES=0 python main_dpt.py \
    --mode infer \
    --exp_name ${EXP_NAME} \
    --iter ${ITER} \
    --data_roots ${DATA_ROOTS[@]} \
    --data_jsons ${TEST_JSONS[@]} \
    --stats_file ${STATS_FILE} \
    --style_enc_ckpt ${STYLE_ENC_CKPT} \
    --output_dir ${OUTPUT_DIR} \
    --n_repetitions ${N_REPETITIONS}"

# 显示完整命令
echo -e "${YELLOW}执行推理命令:${NC}"
echo "$INFER_CMD"
echo ""

# 询问确认
read -p "是否开始推理? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}推理已取消${NC}"
    exit 0
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 开始推理
echo -e "${GREEN}开始推理...${NC}"
echo -e "${GREEN}============================================================${NC}"

eval $INFER_CMD

INFER_EXIT_CODE=$?

if [ $INFER_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}推理完成！${NC}"
    echo -e "${GREEN}输出目录: ${OUTPUT_DIR}${NC}"
    echo -e "${GREEN}============================================================${NC}"
    
    # 显示输出结构
    echo -e "${YELLOW}输出文件结构:${NC}"
    find "${OUTPUT_DIR}" -name "*.npy" -type f 2>/dev/null | head -10 || echo "未找到.npy文件"
    find "${OUTPUT_DIR}" -name "metrics_report.json" -type f 2>/dev/null || echo "未找到metrics_report.json"
    
    # 验证输出结构
    if command -v python &> /dev/null; then
        echo -e "${YELLOW}验证输出结构...${NC}"
        python -c "
import json
from pathlib import Path
output_dir = Path('${OUTPUT_DIR}')
if (output_dir / 'MEAD_VHAP' / 'metrics_report.json').exists():
    print('✓ MEAD_VHAP metrics_report.json 存在')
if (output_dir / 'MultiModal200' / 'metrics_report.json').exists():
    print('✓ MultiModal200 metrics_report.json 存在')
" 2>/dev/null || echo -e "${YELLOW}输出结构验证跳过${NC}"
    fi
else
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}推理失败，退出码: $INFER_EXIT_CODE${NC}"
    echo -e "${RED}============================================================${NC}"
    exit $INFER_EXIT_CODE
fi
