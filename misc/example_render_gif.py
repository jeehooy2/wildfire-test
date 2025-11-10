"""
wildfire-environment 렌더링 및 GIF 저장 예제

이 스크립트는 wildfire 환경을 시뮬레이션하고 결과를 GIF 애니메이션으로 저장합니다.
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
    print("Wildfire Environment GIF 생성 시작")
    print("=" * 60)
    
    # 환경 생성
    env = gym.make("wildfire-v0", 
        num_agents=2,
        max_steps=9000,
        size=17,
        initial_fire_size=3,
        cooperative_reward=False,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[7, 13],
        selfish_region_xmax=[9, 15],
        selfish_region_ymin=[7, 1],
        selfish_region_ymax=[9, 3],
        render_mode="rgb_array",  # 중요: rgb_array 모드로 설정
        render_selfish_region_boundaries=True,  # 이기적 영역 경계 표시
    )
    
    print(f"에이전트 수: {env.num_agents}")
    print(f"그리드 크기: {env.width} x {env.height}")
    print(f"최대 스텝 수: {env.max_steps}")
    print("=" * 60)
    
    # 환경 초기화
    observation, info = env.reset(seed=42)
    
    # 프레임 저장 리스트
    frames = []
    
    # 첫 프레임 렌더링
    frame = env.render()
    frames.append(frame)
    
    print("시뮬레이션 진행 중...")
    
    # 한 에피소드만 실행 (너무 많은 프레임은 파일 크기가 커짐)
    max_steps = 9000
    for step in range(max_steps):
        # 무작위 액션 샘플링
        action = env.action_space.sample()
        
        # 환경 스텝 실행
        observation, reward, done, info = env.step(action)
        
        # 프레임 렌더링 및 저장
        frame = env.render()
        frames.append(frame)
        
        # 에피소드 종료 시 중단
        if done:
            print(f"✓ 에피소드 완료 (스텝: {step + 1})")
            break
    
    env.close()
    
    print(f"총 {len(frames)}개의 프레임 수집 완료")
    print("=" * 60)
    
    # GIF로 저장 (Pillow 사용 - 권장)
    print("\nGIF 저장 중...")
    try:
        save_as_gif_pillow(
            frames, 
            filename="wildfire_simulation.gif",
            fps=10  # 초당 10프레임
        )
    except ImportError:
        print("⚠ Pillow가 설치되지 않았습니다.")
        print("  설치: .venv/bin/pip install pillow")
        print("\nmatplotlib 방식으로 시도 중... (imagemagick 필요)")
        try:
            save_as_gif_matplotlib(
                frames,
                filename="wildfire_simulation.gif",
                fps=10
            )
        except Exception as e:
            print(f"✗ matplotlib 방식도 실패: {e}")
            print("  imagemagick 설치 필요: brew install imagemagick (macOS)")
    
    print("=" * 60)
    print("완료! wildfire_simulation.gif 파일을 확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()

