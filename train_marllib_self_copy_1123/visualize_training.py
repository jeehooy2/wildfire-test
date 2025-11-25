"""
학습 과정 시각화 스크립트

사용법:
python train_marllib_self/visualize_training.py --exp-path run13_fixed
python train_marllib_self/visualize_training.py --exp-path run11 --plot-type dashboard
python train_marllib_self/visualize_training.py --exp-path run15_partial_obs --plot-type reward
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_progress_file(exp_path):
    """experiments/mappo/[exp_name] 경로에서 progress.csv 찾기"""
    progress_files = list(Path(exp_path).rglob("progress.csv"))

    if not progress_files:
        return None

    # 가장 먼저 생성된 progress.csv 반환
    return progress_files[0]


def load_progress_data(progress_file):
    """progress.csv 파일 로드"""
    try:
        df = pd.read_csv(progress_file)
        print(f"✓ 데이터 로드됨: {len(df)} iterations")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None


def plot_episode_reward(df, save_dir=None):
    """에피소드 보상 시각화"""
    plt.figure(figsize=(14, 7))

    if 'episode_reward_mean' in df.columns:
        plt.plot(df.index, df['episode_reward_mean'],
                color='#2E86AB', linewidth=2.5, marker='o', markersize=4, label='Mean Reward')

        # 최대/최소 구간 표시
        if 'episode_reward_max' in df.columns and 'episode_reward_min' in df.columns:
            plt.fill_between(df.index, df['episode_reward_min'], df['episode_reward_max'],
                           alpha=0.2, color='#2E86AB', label='Max/Min Range')

    plt.xlabel('Training Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Episode Reward', fontsize=12, fontweight='bold')
    plt.title('Episode Reward Mean during Training', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "01_episode_reward.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장됨: {save_path}")

    plt.show()


def plot_policy_loss(df, save_dir=None):
    """정책 손실 시각화"""
    plt.figure(figsize=(14, 7))

    # helicopter policy loss
    heli_cols = [col for col in df.columns if 'policy_helicopter' in col and 'policy_loss' in col]
    # truck policy loss
    truck_cols = [col for col in df.columns if 'policy_truck' in col and 'policy_loss' in col]

    if heli_cols:
        plt.plot(df.index, df[heli_cols[0]], color='#A23B72', linewidth=2.5,
                marker='s', markersize=4, label='Helicopter Policy Loss', alpha=0.8)

    if truck_cols:
        plt.plot(df.index, df[truck_cols[0]], color='#F18F01', linewidth=2.5,
                marker='^', markersize=4, label='Truck Policy Loss', alpha=0.8)

    plt.xlabel('Training Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Policy Loss', fontsize=12, fontweight='bold')
    plt.title('Policy Loss during Training', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "02_policy_loss.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장됨: {save_path}")

    plt.show()


def plot_value_loss(df, save_dir=None):
    """가치 함수 손실 시각화"""
    plt.figure(figsize=(14, 7))

    # helicopter vf loss
    heli_cols = [col for col in df.columns if 'policy_helicopter' in col and 'vf_loss' in col]
    # truck vf loss
    truck_cols = [col for col in df.columns if 'policy_truck' in col and 'vf_loss' in col]

    if heli_cols:
        plt.plot(df.index, df[heli_cols[0]], color='#A23B72', linewidth=2.5,
                marker='s', markersize=4, label='Helicopter Value Loss', alpha=0.8)

    if truck_cols:
        plt.plot(df.index, df[truck_cols[0]], color='#F18F01', linewidth=2.5,
                marker='^', markersize=4, label='Truck Value Loss', alpha=0.8)

    plt.xlabel('Training Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Value Function Loss', fontsize=12, fontweight='bold')
    plt.title('Value Function Loss during Training', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "03_value_loss.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장됨: {save_path}")

    plt.show()


def plot_timesteps(df, save_dir=None):
    """타임스텝 진행 시각화"""
    plt.figure(figsize=(14, 7))

    if 'timesteps_total' in df.columns:
        plt.plot(df.index, df['timesteps_total'],
                color='#06A77D', linewidth=2.5, marker='D', markersize=4)

    plt.xlabel('Training Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Total Timesteps', fontsize=12, fontweight='bold')
    plt.title('Total Timesteps Collected', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "04_timesteps.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장됨: {save_path}")

    plt.show()


def plot_policy_rewards(df, save_dir=None):
    """정책별 보상 시각화"""
    plt.figure(figsize=(14, 7))

    heli_mean = [col for col in df.columns if 'policy_helicopter' in col and 'reward_mean' in col]
    truck_mean = [col for col in df.columns if 'policy_truck' in col and 'reward_mean' in col]

    if heli_mean:
        plt.plot(df.index, df[heli_mean[0]], color='#A23B72', linewidth=2.5,
                marker='s', markersize=4, label='Helicopter Reward', alpha=0.8)

    if truck_mean:
        plt.plot(df.index, df[truck_mean[0]], color='#F18F01', linewidth=2.5,
                marker='^', markersize=4, label='Truck Reward', alpha=0.8)

    plt.xlabel('Training Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Policy Reward Mean', fontsize=12, fontweight='bold')
    plt.title('Policy Reward Mean during Training', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "05_policy_rewards.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장됨: {save_path}")

    plt.show()


def plot_combined_dashboard(df, save_dir=None):
    """통합 대시보드 (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Episode Reward
    ax = axes[0, 0]
    if 'episode_reward_mean' in df.columns:
        ax.plot(df.index, df['episode_reward_mean'], color='#2E86AB',
               linewidth=2, marker='o', markersize=3, label='Mean')
        if 'episode_reward_max' in df.columns and 'episode_reward_min' in df.columns:
            ax.fill_between(df.index, df['episode_reward_min'], df['episode_reward_max'],
                           alpha=0.15, color='#2E86AB')
    ax.set_xlabel('Training Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax.set_title('Episode Reward Mean', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 2. Timesteps
    ax = axes[0, 1]
    if 'timesteps_total' in df.columns:
        ax.plot(df.index, df['timesteps_total'], color='#06A77D',
               linewidth=2, marker='D', markersize=3)
    ax.set_xlabel('Training Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Timesteps', fontsize=11, fontweight='bold')
    ax.set_title('Training Progress (Timesteps)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 3. Policy Loss
    ax = axes[1, 0]
    heli_cols = [col for col in df.columns if 'policy_helicopter' in col and 'policy_loss' in col]
    truck_cols = [col for col in df.columns if 'policy_truck' in col and 'policy_loss' in col]
    if heli_cols:
        ax.plot(df.index, df[heli_cols[0]], color='#A23B72', linewidth=2,
               marker='s', markersize=3, label='Helicopter', alpha=0.8)
    if truck_cols:
        ax.plot(df.index, df[truck_cols[0]], color='#F18F01', linewidth=2,
               marker='^', markersize=3, label='Truck', alpha=0.8)
    ax.set_xlabel('Training Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Policy Loss', fontsize=11, fontweight='bold')
    ax.set_title('Policy Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 4. Value Loss
    ax = axes[1, 1]
    heli_cols = [col for col in df.columns if 'policy_helicopter' in col and 'vf_loss' in col]
    truck_cols = [col for col in df.columns if 'policy_truck' in col and 'vf_loss' in col]
    if heli_cols:
        ax.plot(df.index, df[heli_cols[0]], color='#A23B72', linewidth=2,
               marker='s', markersize=3, label='Helicopter', alpha=0.8)
    if truck_cols:
        ax.plot(df.index, df[truck_cols[0]], color='#F18F01', linewidth=2,
               marker='^', markersize=3, label='Truck', alpha=0.8)
    ax.set_xlabel('Training Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value Loss', fontsize=11, fontweight='bold')
    ax.set_title('Value Function Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "00_training_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장됨: {save_path}")

    plt.show()


def print_summary(df, exp_name):
    """학습 요약 정보 출력"""
    print("\n" + "="*80)
    print(f"학습 결과 요약: {exp_name}")
    print("="*80)
    print(f"\n📊 총 반복: {len(df)}")

    if 'episode_reward_mean' in df.columns:
        reward_mean = df['episode_reward_mean'].mean()
        reward_max = df['episode_reward_mean'].max()
        reward_min = df['episode_reward_mean'].min()
        reward_final = df['episode_reward_mean'].iloc[-1]
        print(f"\n🎯 에피소드 보상:")
        print(f"   - 평균: {reward_mean:.4f}")
        print(f"   - 최대: {reward_max:.4f}")
        print(f"   - 최소: {reward_min:.4f}")
        print(f"   - 최종: {reward_final:.4f}")

    if 'timesteps_total' in df.columns:
        total_steps = df['timesteps_total'].iloc[-1]
        print(f"\n⏱️  총 타임스텝: {total_steps:.0f}")

    # 정책별 보상
    heli_mean = [col for col in df.columns if 'policy_helicopter' in col and 'reward_mean' in col]
    truck_mean = [col for col in df.columns if 'policy_truck' in col and 'reward_mean' in col]

    if heli_mean:
        heli_reward = df[heli_mean[0]].iloc[-1]
        print(f"\n🚁 Helicopter Policy 최종 보상: {heli_reward:.4f}")

    if truck_mean:
        truck_reward = df[truck_mean[0]].iloc[-1]
        print(f"🚚 Truck Policy 최종 보상: {truck_reward:.4f}")

    # 손실 정보
    heli_loss = [col for col in df.columns if 'policy_helicopter' in col and 'policy_loss' in col]
    truck_loss = [col for col in df.columns if 'policy_truck' in col and 'policy_loss' in col]

    if heli_loss:
        heli_l = df[heli_loss[0]].iloc[-1]
        print(f"\n📉 Helicopter Policy 최종 손실: {heli_l:.4f}")

    if truck_loss:
        truck_l = df[truck_loss[0]].iloc[-1]
        print(f"📉 Truck Policy 최종 손실: {truck_l:.4f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize MAPPO training results')
    parser.add_argument('--exp-path', type=str, required=True,
                       help='Experiment folder name in experiments/maa2c/ (e.g., run13_fixed)')
    parser.add_argument('--plot-type', type=str, default='all',
                       choices=['all', 'reward', 'policy_loss', 'value_loss', 'timesteps', 'policy_rewards', 'dashboard'],
                       help='Type of plot to generate')
    args = parser.parse_args()

    # 경로 설정
    mappo_dir = Path(__file__).parent / "experiments" / "maa2c"
    exp_dir = mappo_dir / args.exp_path
    save_dir = Path(__file__).parent / "training_progress" / args.exp_path

    # 경로 확인
    if not exp_dir.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {exp_dir}")
        print(f"\n📁 사용 가능한 실험:")
        if mappo_dir.exists():
            for item in sorted(mappo_dir.iterdir()):
                if item.is_dir():
                    print(f"   - {item.name}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Experiment 디렉토리: {exp_dir}")
    print(f"💾 그래프 저장 위치: {save_dir}\n")

    # progress.csv 파일 찾기
    progress_file = find_progress_file(exp_dir)

    if progress_file is None:
        print(f"❌ progress.csv 파일을 찾을 수 없습니다.")
        return

    print(f"📊 Progress 파일: {progress_file}")

    # 데이터 로드
    df = load_progress_data(progress_file)

    if df is None:
        return

    # 요약 출력
    print_summary(df, args.exp_path)

    # 시각화 생성
    print(f"\n📈 그래프 생성 중...")

    if args.plot_type in ['all', 'dashboard']:
        print("   - Training Dashboard...")
        plot_combined_dashboard(df, save_dir)

    if args.plot_type in ['all', 'reward']:
        print("   - Episode Reward...")
        plot_episode_reward(df, save_dir)

    if args.plot_type in ['all', 'policy_loss']:
        print("   - Policy Loss...")
        plot_policy_loss(df, save_dir)

    if args.plot_type in ['all', 'value_loss']:
        print("   - Value Loss...")
        plot_value_loss(df, save_dir)

    if args.plot_type in ['all', 'timesteps']:
        print("   - Timesteps...")
        plot_timesteps(df, save_dir)

    if args.plot_type in ['all', 'policy_rewards']:
        print("   - Policy Rewards...")
        plot_policy_rewards(df, save_dir)

    print(f"\n✨ 시각화 완료!")
    print(f"저장 위치: {save_dir}")


if __name__ == '__main__':
    main()
