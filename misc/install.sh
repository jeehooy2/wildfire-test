#!/bin/bash
# Wildfire Environment 자동 설치 스크립트

set -e  # 오류 발생 시 스크립트 중단

echo "============================================================"
echo "Wildfire Environment 설치 시작"
echo "============================================================"
echo ""

# Python 버전 확인
echo "1. Python 버전 확인 중..."
if command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo "   ✓ Python 3.9 발견"
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
    echo "   ✓ Python 3.8 발견"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$PYTHON_VERSION" = "3.9" ] || [ "$PYTHON_VERSION" = "3.8" ]; then
        PYTHON_CMD="python3"
        echo "   ✓ Python $PYTHON_VERSION 발견"
    else
        echo "   ✗ Python 3.8 또는 3.9가 필요합니다 (현재: $PYTHON_VERSION)"
        exit 1
    fi
else
    echo "   ✗ Python을 찾을 수 없습니다"
    exit 1
fi
echo ""

# uv 확인
echo "2. uv 패키지 관리자 확인 중..."
if ! command -v uv &> /dev/null; then
    echo "   ✗ uv를 찾을 수 없습니다"
    echo "   uv를 설치하려면: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "   ✓ uv 발견"
echo ""

# 가상 환경 생성
echo "3. 가상 환경 생성 중..."
if [ -d ".venv" ]; then
    echo "   ! .venv 디렉토리가 이미 존재합니다"
    read -p "   기존 환경을 삭제하고 새로 만들까요? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        uv venv --python $PYTHON_CMD
        echo "   ✓ 새 가상 환경 생성 완료"
    else
        echo "   기존 환경을 사용합니다"
    fi
else
    uv venv --python $PYTHON_CMD
    echo "   ✓ 가상 환경 생성 완료"
fi
echo ""

# pip 설치
echo "4. pip 설치 중..."
uv pip install pip
echo "   ✓ pip 설치 완료"
echo ""

# 호환 가능한 도구 버전 설치
echo "5. 호환 가능한 빌드 도구 설치 중..."
.venv/bin/pip install "pip<24.1" "setuptools<66" "wheel==0.38.4" --quiet
echo "   ✓ pip $(. .venv/bin/activate && pip --version | cut -d' ' -f2) 설치됨"
echo "   ✓ setuptools, wheel 설치 완료"
echo ""

# 의존성 설치
echo "6. 프로젝트 의존성 설치 중..."
.venv/bin/pip install -r requirements.txt --quiet
echo "   ✓ gym 0.21.0 설치 완료"
echo "   ✓ numpy $(. .venv/bin/activate && python -c 'import numpy; print(numpy.__version__)') 설치 완료"
echo "   ✓ matplotlib 설치 완료"
echo ""

# wildfire-environment 설치
echo "7. wildfire-environment 패키지 설치 중..."
.venv/bin/pip install -e . --quiet
echo "   ✓ wildfire-environment 0.1.9 설치 완료"
echo ""

# 설치 확인
echo "8. 설치 확인 중..."
if .venv/bin/python -c "import gym; import wildfire_environment; env = gym.make('wildfire-v0', num_agents=2, size=17); env.close()" 2>/dev/null; then
    echo "   ✓ 환경이 정상적으로 작동합니다"
else
    echo "   ✗ 환경 테스트 실패"
    exit 1
fi
echo ""

echo "============================================================"
echo "설치가 완료되었습니다!"
echo "============================================================"
echo ""
echo "다음 명령어로 환경을 활성화할 수 있습니다:"
echo "  source .venv/bin/activate"
echo ""
echo "예제를 실행하려면:"
echo "  .venv/bin/python example.py"
echo ""
echo "또는 환경 활성화 후:"
echo "  source .venv/bin/activate"
echo "  python example.py"
echo ""

