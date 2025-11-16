#!/bin/bash
# MARLlib 학습 결과 비교 시각화 예시 명령어들
# train_marllib_self/experiments/mappo/run1의 체크포인트 사용

echo "=========================================="
echo "MARLlib Comparison Visualization Examples"
echo "=========================================="
echo ""

# 프로젝트 루트로 이동
cd /home/bmkim88/wildfire_environment

# 예시 1: checkpoint_000001 사용 (checkpoint-1)
echo "예시 1: checkpoint_000001 (iteration 1) 사용"
echo "----------------------------------------------"
echo "python train_marllib_self/new_compare_visual2.py \\"
echo "  --checkpoint train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03/checkpoint_000001 \\"
echo "  --checkpoint-num 1 \\"
echo "  --episodes 3 \\"
echo "  --seed 42"
echo ""

# 예시 2: checkpoint_000002 사용 (checkpoint-2)
echo "예시 2: checkpoint_000002 (iteration 2) 사용"
echo "----------------------------------------------"
echo "python train_marllib_self/new_compare_visual2.py \\"
echo "  --checkpoint train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03/checkpoint_000002 \\"
echo "  --checkpoint-num 2 \\"
echo "  --episodes 3 \\"
echo "  --seed 42"
echo ""

# 예시 3: 더 많은 에피소드 생성 (5개)
echo "예시 3: 5개 에피소드 생성 (checkpoint_000002 사용)"
echo "----------------------------------------------"
echo "python train_marllib_self/new_compare_visual2.py \\"
echo "  --checkpoint train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03/checkpoint_000002 \\"
echo "  --checkpoint-num 2 \\"
echo "  --episodes 5 \\"
echo "  --seed 0"
echo ""

# 예시 4: 출력 디렉토리 직접 지정
echo "예시 4: 출력 디렉토리 직접 지정"
echo "----------------------------------------------"
echo "python train_marllib_self/new_compare_visual2.py \\"
echo "  --checkpoint train_marllib_self/experiments/mappo/run1/mappo_mlp_wildfire-ma/MAPPOTrainer_wildfire-ma_wildfire-ma_16fa4_00000_0_2025-11-16_05-07-03/checkpoint_000002 \\"
echo "  --checkpoint-num 2 \\"
echo "  --episodes 3 \\"
echo "  --seed 42 \\"
echo "  --output-dir train_marllib_self/results/visualizations/run1_comparison"
echo ""

echo "=========================================="
echo "실행 방법:"
echo "1. 위의 명령어를 복사해서 직접 실행"
echo "2. 또는 이 스크립트를 편집해서 원하는 명령어를 실행"
echo "=========================================="
