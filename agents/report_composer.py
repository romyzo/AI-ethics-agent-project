#!/usr/bin/env python3
"""
Report Composer Agent - Python Version (Markdown → PDF 자동 변환 포함)
최종 AI 윤리 진단 보고서를 생성하고 PDF로 저장합니다.
"""

import os
import sys
import json
from datetime import date
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pypandoc

# 환경 변수 로드
load_dotenv()


# ---------------------------------------------------------
# 1️⃣ 실행 함수 (Supervisor 호출)
# ---------------------------------------------------------
def report_composer_execute():
    """Report Composer 실행 함수"""
    print("\n" + "=" * 60)
    print("📝 Report Composer 시작...")
    print("=" * 60)

    sys.path.append("..")
    from state_manager import load_state, save_state, update_status, save_report_to_file

    state = load_state()

    # Mitigation Recommender 결과 확인
    if state.get("status", {}).get("mitigation_recommender") != "completed":
        print("❌ Mitigation Recommender가 완료되지 않았습니다.")
        return state

    # 보고서 생성
    updated_state = compose_report_and_update_state(state)

    # Markdown 파일 저장
    service_name = (
        updated_state.get("service_profile", {}).get("service_name", "AI_Service")
    )
    report_content = updated_state.get("final_report_markdown", "")

    if not report_content:
        print("❌ Markdown 보고서 내용이 비어 있습니다. PDF 변환 불가.")
        return updated_state

    md_path = save_report_to_file(report_content, service_name)

    # PDF 변환 수행
    pdf_path = convert_markdown_to_pdf(md_path)

    # State에 저장
    updated_state["report_file_path"] = md_path
    updated_state["report_pdf_path"] = pdf_path

    # 상태 저장
    save_state(updated_state)
    update_status(updated_state, "report_composer", "completed")

    print(f"✅ Markdown 및 PDF 보고서 저장 완료!\n📄 {pdf_path}")
    return updated_state


# ---------------------------------------------------------
# 2️⃣ 보고서 작성 함수 (LLM 호출)
# ---------------------------------------------------------
def compose_report_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """State의 모든 정보를 취합하여 최종 Markdown 보고서를 생성하고 State에 저장합니다."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # state.py 구조에 맞게 key 이름 통일
    service_profile = state.get("service_profile", {})
    risk_assessment = state.get("assessment_result", {})
    recommendations = state.get("recommendation_result", [])
    collected_evidence = state.get("evidence_data", {})

    # 필수 데이터 검사
    if not all([service_profile, risk_assessment, recommendations, collected_evidence]):
        print("❌ State에 필수 데이터(Profile, Evidence, Assessment, Recommendation)가 부족합니다.")
        state["report_status"] = "Error: Missing upstream data."
        return state

    print("🧠 Report Composer: LLM 기반 Markdown 생성 중...")

    # 증거 리스트 정리
    evidences_summary = []
    if collected_evidence:
        # baseline_sources와 issue_sources를 합쳐서 처리
        baseline_sources = collected_evidence.get("baseline_sources", [])
        issue_sources = collected_evidence.get("issue_sources", [])
        
        for e in baseline_sources + issue_sources:
            evidences_summary.append({
                "source": e.get("source", "N/A"),
                "category": e.get("category", "기타"),
                "summary": e.get("summary", ""),
            })

    # 프롬프트 정의
    REPORT_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 AI 윤리 진단 보고서를 작성하는 전문가입니다. "
                "제공된 데이터를 기반으로 완전한 Markdown 보고서를 작성하세요. "
                "추가 설명 없이 순수 Markdown 텍스트만 반환해야 합니다.\n\n"
                "[보고서 목차]:\n"
                "1. EXECUTIVE SUMMARY\n"
                "2. SERVICE PROFILE\n"
                "3. RISK ASSESSMENT\n"
                "4. MITIGATION RECOMMENDATIONS\n"
                "5. CONCLUSION\n"
                "6. REFERENCE\n"
                "7. APPENDIX"
            ),
            (
                "human",
                "--- 최종 보고서 생성을 위한 데이터 ---"
                "\n작성일: {today_date}"
                "\n\n[서비스 프로파일]\n{service_profile}"
                "\n\n[리스크 평가 결과]\n{risk_assessment}"
                "\n\n[개선 권고안]\n{recommendations}"
                "\n\n[수집된 증거]\n{evidences_summary}"
            ),
        ]
    )

    chain = REPORT_PROMPT | llm

    try:
        final_report_markdown = chain.invoke(
            {
                "today_date": date.today().isoformat(),
                "service_profile": json.dumps(
                    service_profile, indent=2, ensure_ascii=False
                ),
                "risk_assessment": json.dumps(
                    risk_assessment, indent=2, ensure_ascii=False
                ),
                "recommendations": json.dumps(
                    recommendations, indent=2, ensure_ascii=False
                ),
                "evidences_summary": json.dumps(
                    evidences_summary, indent=2, ensure_ascii=False
                ),
            }
        ).content
    except Exception as e:
        final_report_markdown = f"## ❌ 보고서 생성 실패\n\nLLM 호출 중 오류 발생: {e}"
        state["report_status"] = "Error generating report."
        return state

    # State 저장
    state["final_report_markdown"] = final_report_markdown
    state["report_status"] = "Success"
    print("✅ Report Composer 완료 및 State 업데이트 완료!\n")
    return state


# ---------------------------------------------------------
# 3️⃣ Markdown → PDF 변환 함수
# ---------------------------------------------------------



def convert_markdown_to_pdf(md_path: str) -> str:
    """생성된 Markdown 보고서를 PDF로 자동 변환 (wkhtmltopdf 엔진 사용)"""
    try:
        pdf_path = md_path.replace(".md", ".pdf")
        pypandoc.convert_text(
            open(md_path, "r", encoding="utf-8").read(),
            "pdf",
            format="md",
            outputfile=pdf_path,
            extra_args=["--standalone", "--pdf-engine=wkhtmltopdf"],
        )
        print(f"📄 PDF 변환 완료: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"⚠️ PDF 변환 실패: {e}")
        return ""

# ---------------------------------------------------------
# 4️⃣ 단독 테스트 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    report_composer_execute()
