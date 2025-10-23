#!/usr/bin/env python3
"""
AI 윤리 리스크 진단 에이전트 - 메인 실행 파일
전체 파이프라인을 순차적으로 실행합니다.
"""

import os
import sys

# 에이전트 디렉토리 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🛡️ AI 윤리 리스크 진단 에이전트 시작")
    print("="*80)
    
    # 사용자 입력 받기
    print("\n📝 분석할 AI 서비스에 대해 설명해주세요:")
    print("예시: '채용 지원자의 이력서를 AI로 분석하여 적합한 후보자를 추천하는 시스템입니다. 학력, 경력, 자격증을 바탕으로 직무 적합도를 평가합니다.'")
    
    service_description = input("\n서비스 설명: ")
    
    if not service_description.strip():
        print("❌ 서비스 설명이 입력되지 않았습니다.")
        return
    
    try:
        # 1. Service Profiler 실행
        print("\n" + "="*60)
        print("1️⃣ Service Profiler 실행 중...")
        print("="*60)
        
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
        from service_profiler import service_profiler_execute
        service_profiler_execute(service_description)
        
        # 2. Evidence Collector 실행
        print("\n" + "="*60)
        print("2️⃣ Evidence Collector 실행 중...")
        print("="*60)
        
        from agents.evidence_collector import evidence_collector_execute
        evidence_collector_execute()
        
        # 3. Risk Assessor 실행
        print("\n" + "="*60)
        print("3️⃣ Risk Assessor 실행 중...")
        print("="*60)
        
        from agents.risk_assessor import risk_assessor_execute
        risk_assessor_execute()
        
        # 4. Mitigation Recommender 실행
        print("\n" + "="*60)
        print("4️⃣ Mitigation Recommender 실행 중...")
        print("="*60)
        
        from agents.mitigation_recommender import mitigation_recommender_execute
        mitigation_recommender_execute()
        
        # 5. Report Composer 실행
        print("\n" + "="*60)
        print("5️⃣ Report Composer 실행 중...")
        print("="*60)
        
        from agents.report_composer import report_composer_execute
        final_state = report_composer_execute()
        
        # 최종 결과 출력
        print("\n" + "="*80)
        print("🎉 AI 윤리 리스크 진단 완료!")
        print("="*80)
        
        report_file = final_state.get("report_file_path", "")
        if report_file:
            print(f"📄 최종 보고서가 생성되었습니다: {report_file}")
        else:
            print("📄 보고서 생성에 문제가 있었습니다.")
        
        print("\n✅ 전체 파이프라인 실행 완료!")
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
