#!/usr/bin/env python3
"""
Mitigation Recommender Agent - Python Version
EU AI Act 기준 위험 완화 권고안 생성
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

def mitigation_recommender_execute():
    """Mitigation Recommender 실행 함수"""
    print("\n" + "="*60)
    print("💡 Mitigation Recommender 시작...")
    print("="*60)
    
    # State 로드
    sys.path.append('..')
    from state_manager import load_state, save_state, update_status
    
    state = load_state()
    
    # Risk Assessor 결과 확인
    if state.get("status", {}).get("risk_assessor") != "completed":
        print("❌ Risk Assessor가 완료되지 않았습니다.")
        return state
    
    # 리스크 평가 결과 확인
    assessment_result = state.get("assessment_result", {})
    if not assessment_result:
        print("❌ Risk Assessment 결과가 없습니다.")
        return state
    
    # 권고안 생성 실행
    updated_state = generate_recommendations_and_update_state(state)
    
    # State 저장
    save_state(updated_state)
    update_status(updated_state, "mitigation_recommender", "completed")
    
    print("✅ 권고안 생성 완료")
    
    return updated_state

def generate_recommendations_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """State에서 리스크 평가 결과를 읽고 완화 권고안을 생성한 후, State에 결과를 저장합니다."""
    
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
    
    assessment_result = state.get('assessment_result', {})
    if not assessment_result:
        print("❌ Assessment Result가 없습니다.")
        return state
    
    service_name = assessment_result.get('service_name', 'AI 서비스')
    assessed_risks = assessment_result.get('assessed_risks', [])
    
    if not assessed_risks:
        print("❌ Assessed Risks가 없습니다.")
        return state
    
    print(f"\n💡 Mitigation Recommender 시작: {service_name}의 개선 권고안 생성")
    
    all_recommendations = []
    
    # 권고안 출력 스키마 정의
    RECOMMENDATION_OUTPUT_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "recommendation_title": {"type": "string", "description": "권고안 제목"},
                "mitigation_step": {"type": "string", "description": "구체적인 완화 단계"},
                "priority": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "우선순위"},
                "effort_level": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "구현 노력 수준"},
                "relevant_standard": {"type": "string", "enum": ["EU AI Act", "OECD", "UNESCO"], "description": "관련 표준"}
            },
            "required": ["recommendation_title", "mitigation_step", "priority", "effort_level", "relevant_standard"]
        }
    }
    
    # 권고안 생성 프롬프트
    RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 AI 윤리 리스크 완화 전문가입니다. "
         "제공된 리스크 평가 결과를 바탕으로 구체적이고 실행 가능한 완화 권고안을 생성하세요. "
         "각 권고안은 명확한 제목, 구체적인 실행 단계, 우선순위, 노력 수준, 관련 표준을 포함해야 합니다."
         "\n\n출력 형식: {schema}"),
        ("human", 
         "서비스명: {service_name}\n"
         "리스크 카테고리: {category}\n"
         "리스크 수준: {risk_level}\n"
         "평가 요약: {assessment_summary}\n"
         "권고 초점: {recommendation_focus}\n\n"
         "위 정보를 바탕으로 **{recommendation_focus}**를 해결할 수 있는 구체적인 개선 권고안을 생성하세요. "
         "우선순위(Priority)는 리스크 수준과 연관시켜 설정하세요.")
    ])
    
    for risk in assessed_risks:
        category = risk['category']
        print(f"\n   ⚙️ {category.upper()} 리스크 개선 권고안 생성 중...")
        
        if llm and parser and LANGCHAIN_AVAILABLE:
            try:
                chain = RECOMMENDATION_PROMPT | llm | parser
                
                recommendation_list = chain.invoke({
                    "schema": json.dumps(RECOMMENDATION_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
                    "service_name": service_name,
                    "category": category,
                    "risk_level": risk['risk_level'],
                    "assessment_summary": risk['assessment_summary'],
                    "recommendation_focus": risk['recommendation_focus']
                })
                
                # 카테고리 정보와 함께 리스트에 추가
                for rec in recommendation_list:
                    rec['risk_category'] = category
                
                all_recommendations.extend(recommendation_list)
                print(f"     ✅ 권고안 {len(recommendation_list)}개 생성 완료: {category.upper()}")
                
            except Exception as e:
                print(f"     ❌ 권고안 생성 실패 ({category}): {e}")
                # 간단한 권고안 생성
                simple_recommendation = create_simple_recommendation(category, service_name)
                all_recommendations.extend(simple_recommendation)
        else:
            # 간단한 권고안 생성 (LLM 없이)
            simple_recommendation = create_simple_recommendation(category, service_name)
            all_recommendations.extend(simple_recommendation)
            print(f"     ✅ 권고안 {len(simple_recommendation)}개 생성 완료: {category.upper()}")
    
    # State에 권고안 결과 저장
    state['recommendation_result'] = all_recommendations
    state['recommendation_status'] = "Success"
    
    print(f"\n✅ Mitigation Recommender 완료 및 State 업데이트 완료!")
    print(f"✅ 권고안 생성 완료 - {len(all_recommendations)}개 권고안")
    
    return state

def create_simple_recommendation(category: str, service_name: str) -> List[Dict[str, Any]]:
    """간단한 규칙 기반 권고안 생성 (LLM 없이)"""
    if category == "bias":
        return [
            {
                "recommendation_title": "데이터 다양성 확보",
                "mitigation_step": f"{service_name}에 사용되는 데이터셋의 다양성을 높이기 위해 다양한 출처에서 데이터를 수집하고, 특정 그룹이나 특성에 대한 편향이 없도록 데이터 전처리 과정을 강화합니다.",
                "priority": "High",
                "effort_level": "Medium",
                "relevant_standard": "EU AI Act",
                "risk_category": category
            },
            {
                "recommendation_title": "편향 검출 및 모니터링 시스템 구축",
                "mitigation_step": f"AI 모델의 추천 결과에 대한 편향을 지속적으로 검출하고 모니터링하기 위한 시스템을 구축하여, 정기적으로 결과를 분석하고 필요시 모델을 조정합니다.",
                "priority": "High",
                "effort_level": "High",
                "relevant_standard": "OECD",
                "risk_category": category
            },
            {
                "recommendation_title": "다양성 교육 프로그램 도입",
                "mitigation_step": f"{service_name} 개발 및 운영에 참여하는 팀원들을 대상으로 편향에 대한 인식과 다양성의 중요성을 강조하는 교육 프로그램을 도입하여, 팀원들이 편향 문제를 인식하고 해결할 수 있도록 합니다.",
                "priority": "Medium",
                "effort_level": "Medium",
                "relevant_standard": "UNESCO",
                "risk_category": category
            }
        ]
    elif category == "privacy":
        return [
            {
                "recommendation_title": "데이터 최소화 원칙 적용",
                "mitigation_step": f"사용자 데이터를 수집할 때 필요한 최소한의 정보만 수집하도록 시스템을 설계하고, 불필요한 데이터 수집을 방지하는 정책을 수립합니다.",
                "priority": "High",
                "effort_level": "Medium",
                "relevant_standard": "EU AI Act",
                "risk_category": category
            },
            {
                "recommendation_title": "데이터 암호화 및 익명화",
                "mitigation_step": f"사용자 데이터를 저장하고 전송할 때 강력한 암호화 기술을 적용하고, 데이터 분석 시 개인 식별 정보를 제거하여 익명화하는 절차를 마련합니다.",
                "priority": "High",
                "effort_level": "High",
                "relevant_standard": "OECD",
                "risk_category": category
            },
            {
                "recommendation_title": "사용자 동의 관리 시스템 구축",
                "mitigation_step": f"사용자가 자신의 데이터 수집 및 사용에 대해 명확하게 동의할 수 있도록 하는 시스템을 구축하고, 언제든지 동의를 철회할 수 있는 기능을 제공합니다.",
                "priority": "Medium",
                "effort_level": "Medium",
                "relevant_standard": "UNESCO",
                "risk_category": category
            }
        ]
    else:
        return [
            {
                "recommendation_title": "리스크 관리 체계 구축",
                "mitigation_step": f"{service_name}의 {category} 관련 리스크를 체계적으로 관리하기 위한 정책과 절차를 수립하고, 정기적인 리스크 평가를 수행합니다.",
                "priority": "Medium",
                "effort_level": "Medium",
                "relevant_standard": "EU AI Act",
                "risk_category": category
            }
        ]

if __name__ == "__main__":
    # 테스트 실행
    mitigation_recommender_execute()
