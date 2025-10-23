# step1. 라이브러리 불러오기
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# ✅ agents 폴더를 sys.path에 추가 (절대 경로 사용)
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.join(current_dir, 'agents')
sys.path.insert(0, agents_dir)

print("✅ 라이브러리 불러오기 완료!")
print(f"✅ Agents 폴더 경로: {agents_dir}")
print(f"✅ OpenAI API 키 설정 확인: {'설정됨' if os.getenv('OPENAI_API_KEY') else '설정 안됨'}")

# step2. PDF 파일 업로드 및 서비스 정보 추출
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def extract_service_info_from_pdf(pdf_path):
    """PDF에서 서비스 정보를 추출합니다."""
    try:
        # PDF 로드
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        # 전체 텍스트 결합
        full_text = "\n".join([doc.page_content for doc in docs])
        
        # LLM으로 서비스 정보 추출
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt_template = """다음 PDF 문서에서 AI 서비스에 대한 정보를 추출하여 JSON 형식으로 반환하세요.

PDF 내용:
{content}

추출할 정보:
1. service_name: 서비스 이름
2. service_type: 서비스 유형 (recommendation, classification, prediction 등)
3. description: 서비스 설명 (1-2문장)
4. data_handling_method: 처리하는 데이터 및 처리 방식
5. user_impact_scope: 사용자 영향 범위
6. diagnosed_risk_categories: 리스크 카테고리 (bias, privacy, transparency, accountability, safety, security 중 선택)

JSON 형식으로만 답변하세요."""

        prompt = PromptTemplate.from_template(prompt_template)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"content": full_text})
        
        # 원본 PDF 텍스트도 함께 반환
        result["original_text"] = full_text
        
        return result
        
    except Exception as e:
        print(f"❌ PDF 처리 실패: {e}")
        return None

print("✅ PDF 처리 함수 정의 완료!")

# step3. PDF 파일 경로 설정 및 서비스 정보 추출
print("="*80)
print("🛡️ AI 윤리 리스크 진단 에이전트 시작")
print("="*80)

# PDF 파일 경로 설정 (사용자가 이 부분을 수정하세요)
pdf_file_path = "sample_service.pdf"  # 여기에 실제 PDF 파일 경로를 입력하세요

# PDF 파일이 존재하는지 확인
if not os.path.exists(pdf_file_path):
    print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_file_path}")
    print("📝 사용법:")
    print("1. PDF 파일을 현재 디렉토리에 업로드하세요")
    print("2. 위의 'pdf_file_path' 변수를 실제 파일명으로 수정하세요")
    print("3. 이 셀을 다시 실행하세요")
else:
    print(f"📄 PDF 파일 발견: {pdf_file_path}")
    
    # 서비스 정보 추출
    print("\n🔍 PDF에서 서비스 정보를 추출하는 중...")
    service_info = extract_service_info_from_pdf(pdf_file_path)
    
    if service_info:
        print("✅ 서비스 정보 추출 완료!")
        print(f"   - 서비스명: {service_info.get('service_name', 'N/A')}")
        print(f"   - 서비스 유형: {service_info.get('service_type', 'N/A')}")
        print(f"   - 리스크 카테고리: {', '.join(service_info.get('diagnosed_risk_categories', []))}")
        
        # 추출된 정보를 변수에 저장
        service_description = service_info.get('description', '')
        extracted_service_profile = service_info
        
        print("\n🚀 파이프라인 실행을 시작합니다!")
    else:
        print("❌ 서비스 정보 추출에 실패했습니다.")

# step4. Service Profiler 실행 (PDF에서 추출한 정보 사용)
print("\n" + "="*60)
print("1️⃣ Service Profiler 실행 중...")
print("="*60)

try:
    # PDF에서 추출한 서비스 정보를 사용하여 State에 저장
    from state_manager import load_state, save_state, update_status
    
    # State 로드 및 업데이트
    state = load_state()
    state["service_description"] = service_description
    state["service_profile"] = extracted_service_profile
    save_state(state)
    update_status(state, "service_profiler", "completed")
    
    print("✅ Service Profiler 완료! (PDF 정보를 State에 저장)")
    
except Exception as e:
    print(f"❌ Service Profiler 실행 실패: {e}")
    import traceback
    traceback.print_exc()

# step5. Evidence Collector 실행
print("\n" + "="*60)
print("2️⃣ Evidence Collector 실행 중...")
print("="*60)

try:
    from evidence_collector import evidence_collector_execute
    
    evidence_collector_execute()
    print("✅ Evidence Collector 완료!")
    
except Exception as e:
    print(f"❌ Evidence Collector 실행 실패: {e}")
    import traceback
    traceback.print_exc()

# step6. Risk Assessor 실행
print("\n" + "="*60)
print("3️⃣ Risk Assessor 실행 중...")
print("="*60)

try:
    from risk_assessor import risk_assessor_execute
    
    risk_assessor_execute()
    print("✅ Risk Assessor 완료!")
    
except Exception as e:
    print(f"❌ Risk Assessor 실행 실패: {e}")
    import traceback
    traceback.print_exc()

# step7. Mitigation Recommender 실행
print("\n" + "="*60)
print("4️⃣ Mitigation Recommender 실행 중...")
print("="*60)

try:
    from mitigation_recommender import mitigation_recommender_execute
    
    mitigation_recommender_execute()
    print("✅ Mitigation Recommender 완료!")
    
except Exception as e:
    print(f"❌ Mitigation Recommender 실행 실패: {e}")
    import traceback
    traceback.print_exc()

# step8. Report Composer 실행
print("\n" + "="*60)
print("5️⃣ Report Composer 실행 중...")
print("="*60)

try:
    from report_composer import report_composer_execute
    
    final_state = report_composer_execute()
    print("✅ Report Composer 완료!")
    
except Exception as e:
    print(f"❌ Report Composer 실행 실패: {e}")
    import traceback
    traceback.print_exc()
    final_state = {}

# step9. 최종 결과 출력
print("\n" + "="*80)
print("🎉 AI 윤리 리스크 진단 완료!")
print("="*80)

try:
    report_file = final_state.get("report_file_path", "")
    if report_file:
        print(f"📄 최종 보고서가 생성되었습니다: {report_file}")
    else:
        print("📄 보고서 생성에 문제가 있었습니다.")
    
    print("\n✅ 전체 파이프라인 실행 완료!")
    
except Exception as e:
    print(f"\n❌ 최종 결과 처리 중 오류가 발생했습니다: {e}")