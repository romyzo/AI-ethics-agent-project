#!/usr/bin/env python3
"""
Mitigation Recommender Agent - Python Version
리스크별 개선 권고안 생성
"""

import os
import sys
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

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
    
    assessment_result = state.get("assessment_result", {})
    if not assessment_result:
        print("❌ Assessment Result가 없습니다.")
        return state
    
    # 권고안 생성 실행
    updated_state = generate_recommendations_and_update_state(state)
    
    # State 저장
    save_state(updated_state)
    update_status(updated_state, "mitigation_recommender", "completed")
    
    print(f"✅ 권고안 생성 완료 - {len(updated_state.get('recommendation_result', []))}개 권고안")
    
    return updated_state

def generate_recommendations_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """State에서 평가 결과를 읽고, 개선 권고안을 생성한 후, State에 결과를 저장합니다."""
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    assessment_data = state.get('assessment_result', {})
    if not assessment_data or not assessment_data.get('assessed_risks'):
        state['recommendation_status'] = "Error: Missing assessment_result in state."
        return state
    
    service_name = assessment_data.get('service_name', 'AI 서비스')
    risks_to_mitigate = [
        r for r in assessment_data['assessed_risks'] 
        if r['risk_level'] in ['High', 'Limited'] # High와 Limited 리스크만 권고안 생성
    ]
    
    all_recommendations = []
    parser = JsonOutputParser(pydantic_object=None)
    
    print(f"\n💡 Mitigation Recommender 시작: {service_name}의 개선 권고안 생성")
    
    # JSON 출력 구조 정의
    RECOMMENDATION_OUTPUT_SCHEMA = {
        "recommendation_id": "int (1부터 시작)",
        "recommendation_title": "string (개선 항목의 제목)",
        "mitigation_step": "string (실행 가능한 구체적인 개선 조치 설명)",
        "priority": "string (High, Medium, Low 중 하나)",
        "effort_level": "string (Low, Medium, High 중 하나)",
        "relevant_standard": "string (EU AI Act, OECD, UNESCO 중 가장 관련 높은 기준 명시)"
    }
    
    # 개선 권고 프롬프트 정의
    RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 AI 윤리 리스크 개선 전문가입니다. "
         "제공된 리스크 평가 결과와 '개선 권고 초점'을 기반으로, "
         "구체적이고 실행 가능한 개선 권고안을 3가지 이상 생성하여 JSON 배열 형식으로만 반환하세요. "
         "반드시 출력 JSON 스키마를 준수해야 합니다."
         "\n\n[출력 JSON 스키마]:\n{schema}"
        ),
        ("human", 
         "--- 서비스 및 리스크 평가 정보 ---"
         "\n서비스명: {service_name}"
         "\n평가 리스크 카테고리: {category}"
         "\n위험도 수준: {risk_level}"
         "\n평가 요약: {assessment_summary}"
         "\n--- 개선 권고 초점 ---"
         "\n{recommendation_focus}"
         "\n\n위 정보를 바탕으로 **{recommendation_focus}**를 해결할 수 있는 구체적인 개선 권고안을 생성하세요. "
         "우선순위(Priority)는 리스크 수준과 연관시켜 설정하세요."
        )
    ])
    
    for risk in risks_to_mitigate:
        category = risk['category']
        print(f"\n   ⚙️ {category.upper()} 리스크 개선 권고안 생성 중...")
        
        chain = RECOMMENDATION_PROMPT | llm | parser

        try:
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
            
    # State에 권고안 결과 저장
    state['recommendation_result'] = all_recommendations
    state['recommendation_status'] = "Success"
    
    print("\n✅ Mitigation Recommender 완료 및 State 업데이트 완료!")
    return state

if __name__ == "__main__":
    # 테스트 실행
    mitigation_recommender_execute()