#!/usr/bin/env python3
"""
Risk Assessor Agent - Python Version
EU AI Act 기준 위험 등급 산정
"""

import os
import sys
import json
from typing import Dict, List, Any

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
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain 라이브러리가 설치되지 않았습니다. LLM 기능이 제한됩니다.")

def risk_assessor_execute():
    """Risk Assessor 실행 함수"""
    print("\n" + "="*60)
    print("⚖️ Risk Assessor 시작...")
    print("="*60)
    
    # State 로드
    sys.path.append('..')
    from state_manager import load_state, save_state, update_status
    
    state = load_state()
    
    # Evidence Collector 결과 확인
    if state.get("status", {}).get("evidence_collector") != "completed":
        print("❌ Evidence Collector가 완료되지 않았습니다.")
        return state
    
    evidence_data = state.get("evidence_data", {})
    if not evidence_data:
        print("❌ Evidence Data가 없습니다.")
        return state
    
    # 리스크 평가 실행
    updated_state = assess_risk_and_update_state(state)
    
    # State 저장
    save_state(updated_state)
    update_status(updated_state, "risk_assessor", "completed")
    
    print("✅ 리스크 평가 완료")
    
    return updated_state

def assess_risk_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """State에서 증거 데이터를 읽고 리스크를 평가한 후, State에 결과를 저장합니다."""
    
    # LLM 초기화 (선택적)
    llm = None
    parser = None
    
    if LANGCHAIN_AVAILABLE:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            parser = JsonOutputParser(pydantic_object=None)
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")
            llm = None
            parser = None
    
    evidence_data = state.get('evidence_data', {})
    if not evidence_data:
        print("❌ Evidence Data가 없습니다.")
        return state
    
    service_name = evidence_data.get('query', 'AI 서비스')
    risk_categories = list(evidence_data.get('scores', {}).keys())
    
    # scores가 비어있으면 service_profile에서 risk_categories 가져오기
    if not risk_categories:
        service_profile = state.get('service_profile', {})
        risk_categories = service_profile.get('risk_categories', ['bias', 'privacy', 'transparency'])
        print(f"⚠️ Evidence scores가 비어있어서 service_profile에서 리스크 카테고리를 가져옵니다: {risk_categories}")
    
    assessed_risks = []
    
    print(f"\n⚖️ Risk Assessor 시작: {service_name}의 리스크 평가")
    
    if not risk_categories:
        print("❌ 리스크 카테고리가 없습니다.")
        return state
    
    for category in risk_categories:
        print(f"\n   🔎 {category.upper()} 리스크 평가 중...")
        
        # Baseline 근거 요약 결합
        baseline_summaries = []
        for src in evidence_data.get('baseline_sources', []):
            if src.get('category') == category:
                baseline_summaries.append(f"- [Baseline] 출처: {src['source']} {src['chunk_info']}. 요약: {src['summary']}")
        
        # Issue 근거 요약 결합
        issue_summaries = []
        for src in evidence_data.get('issue_sources', []):
            if src.get('category') == category:
                issue_summaries.append(f"- [Issue] 출처: {src['source']}. 요약: {src['summary']}")
        
        baseline_text = "\n".join(baseline_summaries) if baseline_summaries else "증거 없음 (법적/윤리 기준 미확인 또는 무관)"
        issue_text = "\n".join(issue_summaries) if issue_summaries else "증거 없음 (최신 사회적 이슈 미발견)"

        # 리스크 평가 (선택적)
        if llm and parser and LANGCHAIN_AVAILABLE:
            # 리스크 평가 프롬프트
            ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
                ("system", 
                 "당신은 AI 윤리 리스크를 평가하는 전문가입니다. "
                 "EU AI Act, OECD, UNESCO와 같은 **Baseline 근거**와 **최신 사회 이슈 근거**의 요약을 종합하여, "
                 "제공된 서비스에 대해 특정 리스크 카테고리(편향성, 프라이버시, 투명성 등)의 위험도를 판단하고, 결과를 JSON 형식으로만 반환하세요. "
                 "위험도는 **High, Limited, Minimal** 중 하나여야 합니다."
                 "\n\n[출력 JSON 스키마]:\n{schema}"
                ),
                ("human", 
                 "--- 서비스 및 리스크 정보 ---"
                 "\n서비스명: {service_name}"
                 "\n평가 리스크 카테고리: {category}"
                 "\n--- Baseline 근거 요약 ---"
                 "\n{baseline_summaries}"
                 "\n--- Issue 근거 요약 ---"
                 "\n{issue_summaries}"
                 "\n\n위 정보를 바탕으로 {service_name}의 {category} 리스크 수준을 평가하고, 개선 권고안의 초점을 설정하세요."
                )
            ])

            ASSESSOR_OUTPUT_SCHEMA = {
                "category": "string",
                "risk_level": "string (High, Limited, Minimal 중 택 1)",
                "assessment_summary": "string (평가 근거 및 핵심 이슈를 한국어로 3줄 이내 요약)",
                "recommendation_focus": "string (Mitigation 에이전트가 집중해야 할 구체적인 개선 방향)"
            }

            chain = ASSESSMENT_PROMPT | llm | parser

            try:
                assessment_result = chain.invoke({
                    "schema": json.dumps(ASSESSOR_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
                    "service_name": service_name,
                    "category": category,
                    "baseline_summaries": baseline_text,
                    "issue_summaries": issue_text
                })
                
                if isinstance(assessment_result, dict):
                    assessment_result['category'] = category 
                    assessed_risks.append(assessment_result)
                    print(f"     ✅ 평가 완료: {category.upper()} -> {assessment_result.get('risk_level', 'Unknown')}")
                else:
                    raise Exception("LLM 응답이 딕셔너리가 아닙니다.")
                
            except Exception as e:
                print(f"     ❌ 평가 실패 ({category}): {e}")
                assessment_result = create_simple_assessment(category, service_name)
                assessed_risks.append(assessment_result)
        else:
            # 간단한 리스크 평가 (LLM 없이)
            assessment_result = create_simple_assessment(category, service_name)
            assessed_risks.append(assessment_result)
            print(f"     ✅ 평가 완료: {category.upper()} -> {assessment_result.get('risk_level', 'Unknown')}")

def create_simple_assessment(category: str, service_name: str) -> Dict[str, Any]:
    """간단한 리스크 평가 (LLM 없이)"""
    risk_levels = ["High", "Limited", "Minimal"]
    # 간단한 규칙 기반 평가
    if category == "bias":
        risk_level = "Limited"
    elif category == "privacy":
        risk_level = "Limited"
    else:
        risk_level = "Minimal"
    
    return {
        "category": category,
        "risk_level": risk_level,
        "assessment_summary": f"{service_name}의 {category} 리스크는 EU AI Act 기준으로 {risk_level} 수준으로 평가됩니다.",
        "recommendation_focus": f"{category} 관련 개선 방안 수립"
    }
            
    # 최종 결과 구조화
    final_assessment = {
        "service_name": service_name,
        "assessed_risks": assessed_risks,
    }
    
    # State에 평가 결과 저장
    state['assessment_result'] = final_assessment
    state['assessment_status'] = "Success"
    
    print(f"\n✅ Risk Assessor 평가 및 State 업데이트 완료! (평가된 리스크: {len(assessed_risks)}개)")
    return state

if __name__ == "__main__":
    # 테스트 실행
    risk_assessor_execute()
#!/usr/bin/env python3
"""
Risk Assessor Agent - EU AI Act 기준 위험 등급 산정
"""

import os
import sys
import json
from typing import Dict, Any, List

# ---------------------------------------------------------------------
# 1️⃣ 환경 변수 및 라이브러리 로드
# ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ dotenv 라이브러리가 없습니다. 환경 변수 로딩이 제한됩니다.")

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain이 설치되지 않아 LLM 기반 평가 기능이 비활성화됩니다.")


# ---------------------------------------------------------------------
# 2️⃣ 핵심 실행 함수
# ---------------------------------------------------------------------
def risk_assessor_execute():
    """Risk Assessor 실행 함수"""
    print("\n" + "=" * 60)
    print("⚖️ Risk Assessor 시작...")
    print("=" * 60)

    # state_manager import
    sys.path.append("..")
    from state_manager import load_state, save_state, update_status

    # State 로드
    state = load_state()

    # State가 None이면 기본 구조 생성
    if not isinstance(state, dict) or not state:
        print("⚠️ State가 비어있거나 None입니다. 기본 구조로 초기화합니다.")
        state = {
            "service_profile": {},
            "evidence_data": {},
            "assessment_result": {},
            "status": {}
        }

    # Evidence Collector 완료 여부 확인
    if state.get("status", {}).get("evidence_collector") != "completed":
        print("❌ Evidence Collector가 완료되지 않았습니다.")
        update_status(state, "risk_assessor", "skipped")
        save_state(state)
        return state

    # Evidence Data 확인
    evidence_data = state.get("evidence_data", {})
    if not evidence_data:
        print("⚠️ Evidence Data가 없습니다. 기본 데이터로 진행합니다.")
        evidence_data = {"scores": {"bias": 0.0, "privacy": 0.0}}

    # 평가 수행
    updated_state = assess_risk_and_update_state(state, evidence_data)

    # 결과 저장
    if updated_state:
        update_status(updated_state, "risk_assessor", "completed")
        save_state(updated_state)
        print("✅ 리스크 평가 완료 및 상태 저장 완료!")
        return updated_state
    else:
        print("❌ Risk Assessor 실행 실패 (state 업데이트 불가)")
        update_status(state, "risk_assessor", "failed")
        save_state(state)
        return state


# ---------------------------------------------------------------------
# 3️⃣ 리스크 평가 함수
# ---------------------------------------------------------------------
def assess_risk_and_update_state(state: Dict[str, Any], evidence_data: Dict[str, Any]) -> Dict[str, Any]:
    """증거 데이터를 기반으로 리스크 평가를 수행하고 state에 반영합니다."""

    # 평가 대상
    service_name = evidence_data.get("query", "AI 서비스")
    risk_categories = list(evidence_data.get("scores", {}).keys())

    if not risk_categories:
        service_profile = state.get("service_profile", {})
        risk_categories = service_profile.get("diagnosed_risk_categories", ["bias", "privacy", "transparency"])

    print(f"\n⚖️ Risk Assessor 시작: {service_name}의 리스크 평가")
    assessed_risks: List[Dict[str, Any]] = []

    # 평가용 딕셔너리 초기화
    if "assessment_result" not in state or not isinstance(state["assessment_result"], dict):
        state["assessment_result"] = {}

    # LLM 초기화 (옵션)
    llm = None
    parser = None
    if LANGCHAIN_AVAILABLE:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            parser = JsonOutputParser()
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")
            llm = None

    # 리스크 카테고리별 평가 수행
    for category in risk_categories:
        print(f"\n   🔎 {category.upper()} 리스크 평가 중...")

        if llm and parser:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "EU AI Act, OECD, UNESCO 기준에 따라 리스크를 평가하세요."),
                    ("human", f"서비스명: {service_name}\n리스크 카테고리: {category}\n리스크 수준을 High, Limited, Minimal 중 선택하고 근거를 요약하세요.")
                ])

                chain = prompt | llm | parser
                result = chain.invoke({"service_name": service_name, "category": category})

                if isinstance(result, dict):
                    assessed_risks.append(result)
                    print(f"     ✅ 평가 완료: {category.upper()} -> {result.get('risk_level', 'Unknown')}")
                else:
                    raise Exception("LLM 응답이 딕셔너리가 아닙니다.")
            except Exception as e:
                print(f"     ⚠️ LLM 평가 실패 ({category}): {e}")
                result = create_simple_assessment(category, service_name)
                assessed_risks.append(result)
        else:
            result = create_simple_assessment(category, service_name)
            assessed_risks.append(result)
            print(f"     ✅ 평가 완료: {category.upper()} -> {result['risk_level']}")

    # 평가 결과 정리 및 저장
    final_assessment = {
        "service_name": service_name,
        "assessed_risks": assessed_risks
    }

    state["assessment_result"] = final_assessment
    state["assessment_status"] = "success"

    print(f"\n✅ Risk Assessor 평가 완료! (총 {len(assessed_risks)}개 리스크 평가됨)")
    return state


# ---------------------------------------------------------------------
# 4️⃣ 단순 규칙 기반 평가 (LLM 미사용 시)
# ---------------------------------------------------------------------
def create_simple_assessment(category: str, service_name: str) -> Dict[str, Any]:
    """간단한 규칙 기반 리스크 평가"""
    if category == "bias":
        level = "High"
    elif category == "privacy":
        level = "High"
    else:
        level = "Minimal"

    return {
        "category": category,
        "risk_level": level,
        "assessment_summary": f"{service_name}의 {category} 리스크는 {level} 수준으로 평가되었습니다.",
        "recommendation_focus": f"{category} 관련 개선 방안 수립 필요"
    }


# ---------------------------------------------------------------------
# 5️⃣ 독립 실행 시 테스트
# ---------------------------------------------------------------------
if __name__ == "__main__":
    risk_assessor_execute()
