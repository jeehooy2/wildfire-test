#!/bin/bash
# MARLlib MAPPO 학습 실행 스크립트
#
# 사용법:
#   chmod +x train_rllib/RUN_MARLLIB.sh
#   ./train_rllib/RUN_MARLLIB.sh

set -e  # 에러 발생 시 중단

echo "============================================================"
echo "MARLlib MAPPO (CTDE) 학습 시작"
echo "============================================================"
echo ""

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")/.."

# PYTHONPATH 설정
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Conda 환경 확인
if conda env list | grep -q "marllib-x86"; then
    echo "✓ marllib-x86 환경 발견"
else
    echo "✗ marllib-x86 환경을 찾을 수 없습니다"
    echo "conda create -n marllib-x86 python=3.8로 환경을 먼저 생성하세요"
    exit 1
fi

echo ""
echo "설정 테스트 중..."
echo ""

# 설정 테스트 실행
conda run -n marllib-x86 python train_rllib/test_marllib_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 설정 테스트 실패"
    echo "문제를 해결한 후 다시 시도하세요"
    exit 1
fi

echo ""
echo "============================================================"
echo "학습 시작 준비 완료!"
echo "============================================================"
echo ""
echo "다음 명령으로 학습을 시작하세요:"
echo ""
echo "  # 짧은 테스트 (10 iterations)"
echo "  conda activate marllib-x86"
echo "  python train_rllib/train_marllib_mappo.py --iterations 10"
echo ""
echo "  # 본격적인 학습 (100 iterations)"
echo "  conda activate marllib-x86"
echo "  python train_rllib/train_marllib_mappo.py --iterations 100 --checkpoint-freq 10"
echo ""
echo "  # 긴 학습 (1000 iterations, 실험 이름 지정)"
echo "  conda activate marllib-x86"
echo "  python train_rllib/train_marllib_mappo.py \\"
echo "      --iterations 1000 \\"
echo "      --checkpoint-freq 50 \\"
echo "      --exp-name mappo_long_run"
echo ""
echo "============================================================"
