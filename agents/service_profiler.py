# ==============================================================================

# 👩‍💻 Author    : Hyelim Jo
# 🎯 Purpose   : AI 윤리성 리스크 진단 에이전트 v1.0
# 📅 Created   : 2025-10-22
# 📜 Note      : service_profiler.ipynb

# ==============================================================================
# -------------------------------- Update Log ----------------------------------

# 2025-10-22 14:00 / 초기 생성 / Service Profiler 기본 구조 구현
# 2025-10-22 15:30 / API 키 오류 해결 / load_dotenv() 추가

# ------------------------------------------------------------------------------

# step1. 라이브러리 불러오기
from typing import Dict, Any

# 선택적 라이브러리 import (오류 방지)
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️ dotenv 라이브러리가 설치되지 않았습니다. 환경 변수 로딩이 제한됩니다.")

# 선택적 라이브러리 import (오류 방지)
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain 라이브러리가 설치되지 않았습니다. LLM 기능이 제한됩니다.")

print("[OK] 라이브러리 불러오기 완료")

# step2. 프롬프트 템플릿 정의
if LANGCHAIN_AVAILABLE:
    prompt_template = """당신은 AI 윤리 전문가입니다.

주어진 AI 서비스 설명을 분석하여 다음 정보를 JSON 형식으로 추출해주세요.

# 서비스 설명:
{service_description}

# 추출할 정보:
1. service_name: 서비스 이름
2. service_type: 서비스 유형 (chatbot, recommendation, classification, prediction 중 선택)
3. description: 핵심 설명 (1-2문장)
4. data_processing: 처리하는 데이터
5. user_impact: 사용자 영향
6. risk_categories: 리스크 카테고리 (bias, privacy, transparency, accountability, safety, security 중 선택, 배열)

JSON 형식으로만 답변해주세요.
"""

    prompt = PromptTemplate.from_template(prompt_template)
    print("[OK] 프롬프트 정의 완료")
    
    # step3. LLM 및 Chain 설정
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        json_parser = JsonOutputParser()
        chain = prompt | llm | json_parser
        print("[OK] Chain 구성 완료")
    except Exception as e:
        print(f"⚠️ LLM 초기화 실패: {e}")
        llm = None
        chain = None
else:
    print("⚠️ LangChain이 없어서 간단한 프로파일링을 사용합니다.")
    llm = None
    chain = None
# step4. Service Profiler 함수 정의
def service_profiler(state: Dict[str, Any]) -> Dict[str, Any]:
    """서비스 프로파일 생성"""
    service_description = state.get("service_description", "")
    
    if chain:
        try:
            result = chain.invoke({"service_description": service_description})
        except Exception as e:
            print(f"⚠️ LLM 호출 실패: {e}")
            result = create_simple_profile(service_description)
    else:
        result = create_simple_profile(service_description)
    
    state["service_profile"] = result
    
    print(f"[OK] 프로파일 생성 완료 - {result.get('service_name')} ({result.get('service_type')})")
    return state

def create_simple_profile(service_description: str) -> Dict[str, Any]:
    """간단한 프로파일 생성 (LLM 없이)"""
    return {
        "service_name": "AI 서비스",
        "service_type": "recommendation",
        "description": service_description[:100] + "...",
        "data_processing": "사용자 입력 데이터 처리",
        "user_impact": "일반 사용자",
        "risk_categories": ["bias", "privacy", "transparency"]
    }

print("[OK] 함수 정의 완료")

# step5. State 기반 실행 함수 정의
import sys
sys.path.append('..')
from state_manager import load_state, save_state, update_status

def service_profiler_execute(service_description: str):
    """Service Profiler 실행 함수"""
    print("\n" + "="*60)
    print("🔍 Service Profiler 시작...")
    print("="*60)
    
    # State 로드
    state = load_state()
    
    # 서비스 설명 설정
    state["service_description"] = service_description
    
    # 서비스 프로파일 생성
    if chain:
        try:
            result = chain.invoke({"service_description": service_description})
        except Exception as e:
            print(f"⚠️ LLM 호출 실패: {e}")
            result = create_simple_profile(service_description)
    else:
        result = create_simple_profile(service_description)
    
    state["service_profile"] = result
    
    # State 저장
    save_state(state)
    update_status(state, "service_profiler", "completed")
    
    print(f"✅ 프로파일 생성 완료 - {result.get('service_name')} ({result.get('service_type')})")
    print(f"✅ 리스크 카테고리: {', '.join(result.get('risk_categories', []))}")
    
    return state

print("✅ Service Profiler 실행 함수 정의 완료")

# step6. 테스트
test_state = {
    "service_description": "채용 지원자의 이력서를 AI로 분석하여 적합한 후보자를 추천하는 시스템입니다. 학력, 경력, 자격증을 바탕으로 직무 적합도를 평가합니다.",
    "service_profile": {}
}

result_state = service_profiler(test_state)

print("\n[결과]")
print("서비스명:", result_state["service_profile"]["service_name"])
print("서비스 유형:", result_state["service_profile"]["service_type"])
print("리스크 카테고리:", ", ".join(result_state["service_profile"]["risk_categories"]))
print("\n전체 프로파일:")
print(result_state["service_profile"])

