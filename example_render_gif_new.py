"""
wildfire-environment 여러 종류의 에이전트 렌더링 및 GIF 저장 예제

이 스크립트는 Helicopter, Truck, Crew 등 여러 종류의 에이전트를 사용하여
wildfire 환경을 시뮬레이션하고 결과를 GIF 애니메이션으로 저장합니다.
"""

import gym
import wildfire_environment
from PIL import Image
import os

def save_as_gif_pillow(frames, filename="wildfire_simulation.gif", fps=10):
    """Pillow를 사용하여 프레임을 GIF로 저장

    Parameters
    ----------
    frames : list
        렌더링된 프레임들의 리스트 (numpy arrays)
    filename : str
        저장할 GIF 파일명
    fps : int
        초당 프레임 수 (기본값: 10)
    """
    if not frames:
        print("저장할 프레임이 없습니다!")
        return

    # numpy array를 PIL Image로 변환
    images = [Image.fromarray(frame) for frame in frames]

    # GIF로 저장
    duration = int(1000 / fps)  # 밀리초 단위
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 무한 반복
    )

    print(f"✓ GIF 저장 완료: {filename}")
    print(f"  - 프레임 수: {len(frames)}")
    print(f"  - 파일 크기: {os.path.getsize(filename) / 1024:.1f} KB")


def save_as_gif_matplotlib(frames, filename="wildfire_simulation.gif", fps=10):
    """matplotlib을 사용하여 프레임을 GIF로 저장 (imagemagick 필요)

    Parameters
    ----------
    frames : list
        렌더링된 프레임들의 리스트 (numpy arrays)
    filename : str
        저장할 GIF 파일명
    fps : int
        초당 프레임 수 (기본값: 10)
    """
    from wildfire_environment.utils.misc import save_frames_as_gif

    # 파일명에서 경로와 이름 분리
    path = os.path.dirname(filename) or "./"
    name = os.path.basename(filename).replace(".gif", "")

    save_frames_as_gif(frames, path=path + "/", filename=name, ep=0, fps=fps)
    print(f"✓ GIF 저장 완료: {filename}")


def main():
    print("=" * 60)
    print("Wildfire Environment - 다양한 에이전트 타입 GIF 생성")
    print("=" * 60)

    # 여러 종류의 에이전트를 사용하는 환경 생성
    # Helicopter: 빠르고 효율적 (red)
    # Truck: 중간 속도, 높은 효율 (yellow)
    # Crew: 느리고 낮은 효율 (white)
    num_helicopters = 1
    num_trucks = 1
    num_crews = 1
    total_agents = num_helicopters + num_trucks + num_crews

    # 에이전트 시작 위치 (helicopter, truck, crew 순서)
    agent_start_positions = (
        (1, 1),   # Helicopter 시작 위치
        (15, 1),  # Truck 시작 위치
        (8, 15),  # Crew 시작 위치
    )

    env = gym.make("wildfire-v0",
        num_agents=total_agents,
        num_helicopters=num_helicopters,
        num_trucks=num_trucks,
        num_crews=num_crews,
        agent_start_positions=agent_start_positions,
        max_steps=500,
        size=17,
        initial_fire_size=3,
        cooperative_reward=True,  # 협력 보상
        render_mode="rgb_array",
    )

    print(f"총 에이전트 수: {env.num_agents}")
    print(f"  - Helicopter (red): {num_helicopters}개")
    print(f"  - Truck (yellow): {num_trucks}개")
    print(f"  - Crew (white): {num_crews}개")
    print(f"그리드 크기: {env.width} x {env.height}")
    print(f"최대 스텝 수: {env.max_steps}")
    print(f"초기 불 크기: {env.initial_fire_size}x{env.initial_fire_size}")

    # 에이전트별 특성 출력
    print("\n에이전트 특성:")
    for i, agent in enumerate(env.agents):
        agent_type = agent.__class__.__name__
        speed = agent.speed if hasattr(agent, 'speed') else 1.0
        efficiency = agent.efficiency if hasattr(agent, 'efficiency') else 1.0
        print(f"  Agent {i} ({agent_type}): speed={speed}, efficiency={efficiency}, color={agent.color}")

    print("=" * 60)

    # 환경 초기화
    observation, info = env.reset(seed=42)

    # 프레임 저장 리스트
    frames = []

    # 첫 프레임 렌더링
    frame = env.render()
    frames.append(frame)

    print("\n시뮬레이션 진행 중 (무작위 행동)...")

    # 한 에피소드 실행
    max_steps = 500
    step = 0
    terminated = False
    truncated = False

    while step < max_steps and not (terminated or truncated):
        # 무작위 액션 샘플링
        action = env.action_space.sample()

        # 환경 스텝 실행
        observation, reward, terminated, truncated, info = env.step(action)

        # 프레임 렌더링 및 저장
        frame = env.render()
        frames.append(frame)

        # 진행 상황 출력 (매 50 스텝마다)
        if (step + 1) % 50 == 0:
            print(f"  스텝 {step + 1}/{max_steps}: 불타는 나무={info['0']['trees_on_fire']}, 타버린 나무={info['0']['burnt_trees_total']}")

        step += 1

        # 에피소드 종료 시 중단
        if terminated or truncated:
            print(f"✓ 에피소드 완료 (스텝: {step})")
            print(f"  - 최종 불타는 나무: {info['0']['trees_on_fire']}")
            print(f"  - 최종 타버린 나무: {info['0']['burnt_trees_total']}")
            break

    env.close()

    print(f"\n총 {len(frames)}개의 프레임 수집 완료")
    print("=" * 60)

    # GIF로 저장 (Pillow 사용 - 권장)
    print("\nGIF 저장 중...")
    output_filename = "wildfire_simulation_multi_agent.gif"

    try:
        save_as_gif_pillow(
            frames,
            filename=output_filename,
            fps=10  # 초당 10프레임
        )
    except ImportError:
        print("⚠ Pillow가 설치되지 않았습니다.")
        print("  설치: pip install pillow")
        print("\nmatplotlib 방식으로 시도 중... (imagemagick 필요)")
        try:
            save_as_gif_matplotlib(
                frames,
                filename=output_filename,
                fps=10
            )
        except Exception as e:
            print(f"✗ matplotlib 방식도 실패: {e}")
            print("  imagemagick 설치 필요: brew install imagemagick (macOS)")

    print("=" * 60)
    print(f"완료! {output_filename} 파일을 확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
