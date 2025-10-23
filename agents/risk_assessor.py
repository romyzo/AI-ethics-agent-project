#!/usr/bin/env python3
"""
Risk Assessor Agent - EU AI Act 기준 위험 등급 산정
"""

import os
import sys
import json
import re
from typing import Dict, Any, List

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ dotenv 라이브러리가 없습니다.")

# LangChain 라이브러리 로드
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain이 설치되지 않아 LLM 기반 평가 기능이 비활성화됩니다.")


def risk_assessor_execute():
    """Risk Assessor 실행 함수"""
    print("\n" + "=" * 60)
    print("⚖️ Risk Assessor 시작...")
    print("=" * 60)

    sys.path.append("..")
    from state_manager import load_state, save_state, update_status

    state = load_state()

    # Evidence Collector 완료 여부 확인
    if state.get("status", {}).get("evidence_collector") != "completed":
        print("❌ Evidence Collector가 완료되지 않았습니다. Risk Assessor를 건너뜁니다.")
        update_status(state, "risk_assessor", "skipped")
        save_state(state)
        return state

    # Evidence Data 확인
    evidence_data = state.get("evidence_data", {})
    if not evidence_data:
        print("❌ Evidence Data가 없습니다. Risk Assessor를 건너뜁니다.")
        update_status(state, "risk_assessor", "failed")
        save_state(state)
        return state

    # 평가 수행
    updated_state = assess_risk_and_update_state(state)

    # 결과 저장
    save_state(updated_state)
    update_status(updated_state, "risk_assessor", "completed")
    
    print("✅ 리스크 평가 완료 및 상태 저장 완료!")
    return updated_state


def assess_risk_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """증거 데이터를 기반으로 리스크 평가를 수행하고 state에 반영합니다."""

    evidence_data = state.get("evidence_data", {})
    service_name = evidence_data.get("query", "AI 서비스")
    risk_categories = list(evidence_data.get("scores", {}).keys())

    # Risk categories가 비어있으면 service_profile에서 가져오기
    if not risk_categories:
        service_profile = state.get("service_profile", {})
        risk_categories = service_profile.get("diagnosed_risk_categories", ["bias", "privacy", "transparency"])

    print(f"\n⚖️ Risk Assessor: {service_name}의 리스크 평가 시작")
    assessed_risks = []

    # LLM 초기화
    llm = None
    if LANGCHAIN_AVAILABLE:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")

    # 각 리스크 카테고리별 평가
    for category in risk_categories:
        print(f"\n   🔎 {category.upper()} 리스크 평가 중...")

        # Baseline 근거 수집
        baseline_summaries = []
        for src in evidence_data.get("baseline_sources", []):
            if src.get("category") == category:
                baseline_summaries.append(
                    f"- [Baseline] 출처: {src['source']} {src['chunk_info']}. 요약: {src['summary']}"
                )

        # Issue 근거 수집
        issue_summaries = []
        for src in evidence_data.get("issue_sources", []):
            if src.get("category") == category:
                issue_summaries.append(
                    f"- [Issue] 출처: {src['source']}. 요약: {src['summary']}"
                )

        baseline_text = "\n".join(baseline_summaries) if baseline_summaries else "증거 없음"
        issue_text = "\n".join(issue_summaries) if issue_summaries else "증거 없음"

        # LLM 기반 평가
        if llm:
            try:
                assessment_result = evaluate_with_llm(
                    llm, service_name, category, baseline_text, issue_text
                )
                assessed_risks.append(assessment_result)
                print(f"     ✅ 평가 완료: {category.upper()} -> {assessment_result.get('risk_level', 'Unknown')}")
            except Exception as e:
                print(f"     ⚠️ LLM 평가 실패 ({category}): {e}")
                result = create_simple_assessment(category, service_name)
                assessed_risks.append(result)
        else:
            # 규칙 기반 평가
            result = create_simple_assessment(category, service_name)
            assessed_risks.append(result)
            print(f"     ✅ 평가 완료: {category.upper()} -> {result['risk_level']}")

    # ✅ 핵심: State에 평가 결과 저장
    final_assessment = {
        "service_name": service_name,
        "assessed_risks": assessed_risks
    }

    state["assessment_result"] = final_assessment
    state["assessment_status"] = "success"

    print(f"\n✅ Risk Assessor 평가 완료! (총 {len(assessed_risks)}개 리스크 평가됨)")
    return state


def evaluate_with_llm(llm, service_name, category, baseline_text, issue_text):
    """LLM을 사용한 리스크 평가 - JSON 강제 파싱"""
    
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    
    # ✅ Pydantic 모델로 JSON 구조 강제
    class RiskAssessment(BaseModel):
        category: str = Field(description="리스크 카테고리")
        risk_level: str = Field(description="High, Limited, Minimal 중 하나")
        assessment_summary: str = Field(description="평가 근거 3줄 요약")
        recommendation_focus: str = Field(description="구체적인 개선 방향")
    
    parser = JsonOutputParser(pydantic_object=RiskAssessment)
    
    # ✅ JSON 형식을 강제하는 프롬프트
    ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 AI 윤리 리스크를 평가하는 전문가입니다. "
         "EU AI Act, OECD, UNESCO 기준을 바탕으로 리스크를 평가하세요.\n\n"
         "{format_instructions}"
        ),
        ("human", 
         "서비스명: {service_name}\n"
         "리스크 카테고리: {category}\n\n"
         "Baseline 근거:\n{baseline_summaries}\n\n"
         "Issue 근거:\n{issue_summaries}\n\n"
         "위 정보를 바탕으로 평가하세요."
        )
    ])

    chain = ASSESSMENT_PROMPT | llm

    response = chain.invoke({
        "service_name": service_name,
        "category": category,
        "baseline_summaries": baseline_text,
        "issue_summaries": issue_text
    })

    # ✅ 강제 JSON 파싱
    try:
        # JSON 코드 블록 추출
        content = response.content
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            # JSON 블록 없으면 전체에서 찾기
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("JSON을 찾을 수 없습니다")
        
        result["category"] = category
        return result
        
    except Exception as e:
        print(f"     ⚠️ JSON 파싱 실패, 텍스트에서 정보 추출 시도: {e}")
        # 텍스트에서 정보 추출
        content = response.content
        
        # 리스크 레벨 추출
        if "High" in content:
            risk_level = "High"
        elif "Limited" in content:
            risk_level = "Limited"
        else:
            risk_level = "Minimal"
        
        # 첫 500자를 요약으로 사용
        summary = content[:500].replace("\n", " ")
        
        return {
            "category": category,
            "risk_level": risk_level,
            "assessment_summary": summary,
            "recommendation_focus": f"{category} 관련 개선 방안 수립 필요"
        }


def create_simple_assessment(category: str, service_name: str) -> Dict[str, Any]:
    """간단한 규칙 기반 리스크 평가"""
    risk_mapping = {
        "bias": "High",
        "privacy": "High",
        "transparency": "Limited",
        "accountability": "Limited",
        "safety": "Minimal",
        "security": "Limited"
    }

    level = risk_mapping.get(category, "Limited")

    return {
        "category": category,
        "risk_level": level,
        "assessment_summary": f"{service_name}의 {category} 리스크는 EU AI Act 기준으로 {level} 수준으로 평가됩니다.",
        "recommendation_focus": f"{category} 관련 개선 방안 수립 필요"
    }


if __name__ == "__main__":
    risk_assessor_execute()