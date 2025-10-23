#!/usr/bin/env python3
"""
Report Composer Agent - Markdown → PDF 자동 변환 포함
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

# pypandoc이 설치되어 있는지 확인 (Pandoc이 시스템에 설치되어 있어야 합니다)
try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False
    print("⚠️ pypandoc 라이브러리가 설치되지 않았습니다. PDF 변환 기능이 제한됩니다.")
except OSError as e:
    # Pandoc 실행 파일이 시스템 PATH에 없는 경우
    PYPANDOC_AVAILABLE = False
    print(f"⚠️ Pandoc 실행 파일을 찾을 수 없습니다. PDF 변환이 불가능합니다. (Error: {e})")

# .env 파일 로드
load_dotenv()

# State Manager 모듈 경로 설정
try:
    # agents 폴더 내에서 실행되거나, main.py에서 호출될 때의 경로를 고려하여 상태 관리 모듈 로드
    from state_manager import load_state, save_state, update_status, save_report_to_file
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    # sys.path를 동적으로 추가하여 재시도 (테스트 환경 고려)
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    try:
        from state_manager import load_state, save_state, update_status, save_report_to_file
        STATE_MANAGER_AVAILABLE = True
    except ImportError:
        STATE_MANAGER_AVAILABLE = False
        print("❌ state_manager 모듈을 찾을 수 없습니다. Agent 실행이 불가능합니다.")


def report_composer_execute():
    """Report Composer 실행 함수"""
    print("\n" + "=" * 60)
    print("📝 Report Composer 시작...")
    print("=" * 60)

    if not STATE_MANAGER_AVAILABLE:
        return {}
        
    state = load_state()

    # ✅ 핵심: Mitigation Recommender 완료 여부 확인
    if state.get("status", {}).get("mitigation_recommender") != "completed":
        print("❌ Mitigation Recommender가 완료되지 않았습니다.")
        print("💡 해결 방법: 전체 파이프라인을 처음부터 다시 실행하세요.")
        update_status(state, "report_composer", "skipped")
        save_state(state)
        return state

    # ✅ 필수 데이터 검증
    validation_result = validate_required_data(state)
    if not validation_result["valid"]:
        print(f"❌ 필수 데이터 누락: {', '.join(validation_result['missing'])}")
        print("💡 해결 방법:")
        for agent in validation_result['missing']:
            print(f"   - {agent} 에이전트를 다시 실행하세요")
        
        state["report_status"] = f"Error: Missing data - {', '.join(validation_result['missing'])}"
        update_status(state, "report_composer", "failed")
        save_state(state)
        return state

    # 보고서 생성
    updated_state = compose_report_and_update_state(state)

    # Markdown 파일 저장
    service_name = updated_state.get("service_profile", {}).get("service_name", "AI_Service")
    report_content = updated_state.get("final_report_markdown", "")

    if not report_content:
        print("❌ Markdown 보고서 내용이 비어 있습니다.")
        update_status(updated_state, "report_composer", "failed")
        save_state(updated_state)
        return updated_state

    md_path = save_report_to_file(report_content, service_name)

    # PDF 변환 수행 (오류가 수정된 부분)
    pdf_path = convert_markdown_to_pdf(md_path)

    # State에 저장
    updated_state["report_file_path"] = md_path
    updated_state["report_pdf_path"] = pdf_path

    # 상태 저장
    update_status(updated_state, "report_composer", "completed")
    save_state(updated_state)

    print(f"✅ Markdown 및 PDF 보고서 저장 완료!\n📄 MD: {md_path}\n📄 PDF: {pdf_path if pdf_path else 'PDF 변환 실패'}")
    return updated_state


def validate_required_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """필수 데이터 검증 (가짜 데이터 생성하지 않음)"""
    missing = []
    
    service_profile = state.get("service_profile", {})
    risk_assessment = state.get("assessment_result", {})
    recommendations = state.get("recommendation_result", [])
    collected_evidence = state.get("evidence_data", {})
    
    # Service Profile 체크 (Service Profiler)
    if not service_profile or not service_profile.get("service_name"):
        missing.append("Service Profile (Service Profiler)")
    
    # Risk Assessment 체크 (Risk Assessor)
    if not risk_assessment or not risk_assessment.get("assessed_risks"):
        missing.append("Risk Assessment (Risk Assessor)")
    
    # Recommendations 체크 (Mitigation Recommender)
    if not recommendations or len(recommendations) == 0:
        missing.append("Recommendations (Mitigation Recommender)")
    
    # Evidence Data 체크 (Evidence Collector)
    # 최소한 Baseline (공식 기준) 증거는 있어야 함
    if not collected_evidence or not collected_evidence.get("baseline_sources"):
        missing.append("Evidence Data (Evidence Collector)")
    
    return {
        "valid": len(missing) == 0,
        "missing": missing
    }


def compose_report_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """State의 모든 정보를 취합하여 최종 Markdown 보고서를 생성하고 State에 저장합니다."""

    # LLM 초기화 (Report Composer는 LLM이 필수)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    except Exception as e:
        print(f"❌ LLM 초기화 실패: {e}. 보고서 생성을 건너뜁니다.")
        state["final_report_markdown"] = "## ❌ 보고서 생성 실패\n\nLLM 초기화 오류로 보고서를 생성할 수 없습니다."
        state["report_status"] = "Error: LLM init failed."
        return state


    service_profile = state.get("service_profile", {})
    risk_assessment = state.get("assessment_result", {})
    recommendations = state.get("recommendation_result", [])
    collected_evidence = state.get("evidence_data", {})

    print("🧠 Report Composer: LLM 기반 Markdown 생성 중...")

    # 증거 리스트 정리 (LLM에 전달할 요약본)
    evidences_summary = []
    baseline_sources = collected_evidence.get("baseline_sources", [])
    issue_sources = collected_evidence.get("issue_sources", [])
    
    for e in baseline_sources + issue_sources:
        evidences_summary.append({
            "source": e.get("source", "N/A"),
            "document_type": e.get("document_type", "N/A"),
            "category": e.get("category", "기타"),
            "summary": e.get("summary", ""),
        })

    # 프롬프트 정의
    REPORT_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
          "당신은 기업용 AI 윤리 진단 보고서를 작성하는 전문가입니다. "
          "제공된 데이터를 기반으로 완전한 Markdown 보고서를 작성하세요. "
          "추가 설명 없이 순수 Markdown 텍스트만 반환해야 합니다.\n\n"
          "보고서는 다음 구조로 작성하세요:\n\n"
          "# AI 윤리 리스크 진단 보고서\n"
          "## {service_name}\n\n"
          "**---**\n\n"
          "## 📋 Executive Summary\n"
          "제공된 데이터를 기반으로 3문단으로 요약하여 작성하세요. 전체 리스크 수준, 주요 리스크, 핵심 권고사항을 포함해야 합니다.\n\n"
          "### 주요 내용\n"
          "- **전체 리스크 수준**: [High/Limited/Minimal]\n"
          "- **주요 리스크 카테고리**: [카테고리 목록]\n"
          "- **핵심 권고사항**: [3-5개 핵심 권고사항]\n\n"
          "**---**\n\n"
          "## 🎯 Service Profile\n"
          "제공된 서비스 프로파일 정보를 Markdown 테이블과 리스트를 조합하여 보기 쉽게 정리하세요.\n\n"
          "### 기본 정보\n"
          "- **서비스명**: [이름]\n"
          "- **서비스 유형**: [유형]\n"
          "- **설명**: [상세 설명]\n"
          "- **데이터 처리 방식**: [처리 방식]\n"
          "- **사용자 영향 범위**: [영향 범위]\n"
          "- **진단된 리스크 카테고리**: [카테고리 목록]\n\n"
          "**---**\n\n"
          "## ⚖️ Risk Assessment\n"
          "리스크 평가 결과를 기반으로 각 카테고리별 평가 요약 및 우려 사항을 상세히 기술하세요.\n"
          "### 리스크 평가 결과\n"
          "[각 리스크별로]\n"
          "#### [카테고리명] 리스크\n"
          "- **리스크 수준**: [High/Limited/Minimal]\n"
          "- **평가 요약**: [상세 평가]\n"
          "- **주요 우려사항**: [우려사항 목록]\n"
          "- **권고 초점**: [개선 방향]\n\n"
          "**---**\n\n"
          "## 💡 Mitigation Recommendations\n"
          "개선 권고안을 우선순위별로 구분하여 구체적인 실행 계획과 함께 작성하세요.\n"
          "### 우선순위별 개선 권고안\n"
          "#### 🔴 High Priority\n"
          "[High 우선순위 권고안들]\n\n"
          "#### 🟡 Medium Priority\n"
          "[Medium 우선순위 권고안들]\n\n"
          "#### 🟢 Low Priority\n"
          "[Low 우선순위 권고안들]\n\n"
          "**---**\n\n"
          "## 📚 Evidence Sources\n"
          "증거 수집 결과를 바탕으로 어떤 문서와 이슈가 진단에 사용되었는지 명확히 보여주세요.\n"
          "### Baseline Sources (공식 문서)\n"
          "[EU AI Act, OECD, UNESCO 문서 기반 증거 요약 및 출처]\n\n"
          "### Issue Sources (최신 이슈)\n"
          "[웹 크롤링 기반 최신 이슈 증거 요약 및 출처]\n\n"
          "**---**\n\n"
          "## 📄 Conclusion\n"
          "### 전체 평가\n"
          "진단 결과를 간략하게 요약하고, 이 보고서의 의미를 설명하세요.\n\n"
          "### 권장 다음 단계\n"
          "구체적인 액션 플랜과 담당 부서에 대한 권고 사항을 제시하세요.\n\n"
          "### 연락처 및 지원\n"
          "본 보고서에 대한 문의사항이나 추가 지원이 필요한 경우 관련 담당자에게 연락하시기 바랍니다.\n\n"
          "**---**\n\n"
          "*본 보고서는 EU AI Act, OECD, UNESCO 기준에 따라 작성되었습니다.*\n"
          "*보고서 생성일: {today_date}*"
        ),
        ("human",
          "--- 최종 보고서 생성을 위한 데이터 ---"
          "\n작성일: {today_date}"
          "\n\n[서비스 프로파일]\n{service_profile}"
          "\n\n[리스크 평가 결과]\n{risk_assessment}"
          "\n\n[개선 권고안]\n{recommendations}"
          "\n\n[수집된 증거]\n{evidences_summary}"
        )
    ])

    chain = REPORT_PROMPT | llm

    try:
        final_report_markdown = chain.invoke({
            "today_date": date.today().isoformat(),
            "service_name": service_profile.get("service_name", "AI 서비스"),
            "service_profile": json.dumps(service_profile, indent=2, ensure_ascii=False),
            "risk_assessment": json.dumps(risk_assessment, indent=2, ensure_ascii=False),
            "recommendations": json.dumps(recommendations, indent=2, ensure_ascii=False),
            "evidences_summary": json.dumps(evidences_summary, indent=2, ensure_ascii=False),
        }).content
    except Exception as e:
        final_report_markdown = f"## ❌ 보고서 생성 실패\n\nLLM 호출 중 오류 발생: {e}"
        state["report_status"] = "Error generating report."
        return state

    # State 저장
    state["final_report_markdown"] = final_report_markdown
    state["report_status"] = "Success"
    print("✅ Report Composer 완료 및 State 업데이트 완료!\n")
    return state


def convert_markdown_to_pdf(md_path: str) -> str:
    """생성된 Markdown 보고서를 PDF로 자동 변환 (오류 수정)"""
    if not PYPANDOC_AVAILABLE:
        print("⚠️ pypandoc 또는 Pandoc이 없어서 PDF 변환을 건너뜝니다.")
        return ""
        
    try:
        pdf_path = md_path.replace(".md", ".pdf")
        
        print(f"📄 PDF 변환 시작: {md_path} -> {pdf_path}")
        
        # 💡 오류 수정: wkhtmltopdf에서 인식하지 못하는 --margin-*, --encoding 인자 제거
        pypandoc.convert_text(
            open(md_path, "r", encoding="utf-8").read(),
            "pdf",
            format="md",
            outputfile=pdf_path,
            extra_args=[
                "--standalone",
                "--pdf-engine=wkhtmltopdf", # PDF 엔진 지정
                # CSS를 사용하여 스타일링 (여백은 wkhtmltopdf 기본값 사용)
                "--css=https://cdn.jsdelivr.net/npm/github-markdown-css@5.2.0/github-markdown-light.min.css", 
                # 메타데이터 설정
                "--metadata=title=AI 윤리 리스크 진단 보고서",
                "--metadata=author=AI Ethics Risk Diagnosis System",
                "--metadata=date=" + date.today().strftime("%Y-%m-%d")
            ],
        )
        print(f"✅ PDF 변환 완료: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"⚠️ PDF 변환 실패: Pandoc died or other error: {e}")
        return ""


if __name__ == "__main__":
    report_composer_execute()
