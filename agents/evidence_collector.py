# ==============================================================================

# 👩‍💻 Author    : Hyelim Jo
# 🎯 Purpose   : AI 윤리성 리스크 진단 에이전트 v1.0
# 📅 Created   : 2025-10-22
# 📜 Note      : evidence_collector.ipynb

# ==============================================================================

# -------------------------------- Update Log ----------------------------------

# 2025-10-22 16:00 / 초기 생성 / Evidence Collector 기본 구조 구현
# 2025-10-22 16:30 / RAG 메모리 설계 / Baseline + Issue 메모리 분리
# 2025-10-22 17:00 / HuggingFace 임베딩 적용 / 경제성 개선
# 2025-10-23 09:00 / 웹 크롤링 실제 구현 / Tavily Search API를 사용하여 최신 뉴스/논문 수집
# 2025-10-23 09:30 / Baseline 쿼리 강화 / EU, OECD, UNESCO 기준 명시 및 파일 구성에 맞춰 로드 로직 명확화
# 2025-10-23 11:00 / 평가 로직 구현 / LLM 기반의 위험도(High/Limited/Minimal) 평가
# 2025-10-23 11:30 / JSON 출력 포맷 정의 / Mitigation Recommender에게 전달할 구조 확정

# ------------------------------------------------------------------------------

# step1. 라이브러리 불러오기
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 선택적 라이브러리 import (오류 방지)
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️ BeautifulSoup 라이브러리가 설치되지 않았습니다. 웹 크롤링 기능이 제한됩니다.")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠️ langchain_huggingface 라이브러리가 설치되지 않았습니다. 로컬 임베딩이 제한됩니다.")

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ langchain_chroma 라이브러리가 설치되지 않았습니다. 벡터 저장소가 제한됩니다.")

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    PDF_LOADER_AVAILABLE = True
except ImportError:
    PDF_LOADER_AVAILABLE = False
    print("⚠️ PyMuPDFLoader 라이브러리가 설치되지 않았습니다. PDF 로딩이 제한됩니다.")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
except ImportError:
    TEXT_SPLITTER_AVAILABLE = False
    print("⚠️ RecursiveCharacterTextSplitter 라이브러리가 설치되지 않았습니다.")

try:
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain 라이브러리가 설치되지 않았습니다. LLM 기능이 제한됩니다.")

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️ Tavily 라이브러리가 설치되지 않았습니다. 웹 검색이 제한됩니다.")

print("✅ 라이브러리 불러오기 완료!")

# step2. 설정 및 경로 정의
# 데이터 경로 설정 (agents 폴더 내에서 실행 가정)
base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
reference_dir = os.path.join(base_dir, "reference")
reference_dir = os.path.join(base_dir, "reference")
crawled_dir = os.path.join(base_dir, "crawled")
processed_dir = os.path.join(base_dir, "processed")
baseline_embed_dir = os.path.join(base_dir, "embeddings", "baseline")
issue_embed_dir = os.path.join(base_dir, "embeddings", "issue")

# 디렉토리 생성
for dir_path in [crawled_dir, processed_dir, baseline_embed_dir, issue_embed_dir]:
    os.makedirs(dir_path, exist_ok=True)

# step3. 임베딩 모델 및 LLM 초기화
embedding_model = None
llm = None

# 임베딩 모델 초기화 (선택적)
if HUGGINGFACE_AVAILABLE:
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ HuggingFace 임베딩 모델 초기화 완료!")
    except Exception as e:
        print(f"⚠️ HuggingFace 임베딩 모델 초기화 실패: {e}")
        embedding_model = None

# LLM 초기화 (선택적)
if LANGCHAIN_AVAILABLE:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("✅ ChatOpenAI LLM 초기화 완료!")
    except Exception as e:
        print(f"⚠️ ChatOpenAI 초기화 실패: {e}")
        llm = None
# step4. Baseline 메모리 구축 (EU, OECD, UNESCO 문서)
def build_baseline_memory():
    """공식 문서 기반 Baseline 메모리 구축"""
    baseline_docs = []
    
    # PDF 파일 로드 (선택적)
    pdf_files = [
        "EU_AI_Act.pdf",
        "OECD_Privacy_2024.pdf", 
        "UNESCO_Ethics_2021.pdf"
    ]
    
    if PDF_LOADER_AVAILABLE:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(reference_dir, pdf_file)
            
            if os.path.exists(pdf_path):
                try:
                    loader = PyMuPDFLoader(pdf_path)
                    docs = loader.load()
                    print(f"✅ {pdf_file} 로드 완료")
                    
                    # 메타데이터에 문서 타입 추가 및 페이지 번호 정보 포함
                    for doc in docs:
                        doc.metadata["document_type"] = "baseline"
                        doc.metadata["source"] = pdf_file
                        doc.metadata["page"] = doc.metadata.get("page", 0) + 1 # 페이지 번호는 1부터 시작
                    baseline_docs.extend(docs)
                except Exception as e:
                    print(f"⚠️ {pdf_file} 로드 실패: {e}")
            else:
                print(f"⚠️ {pdf_file} 파일이 지정된 경로에 없습니다: {pdf_path}")
    else:
        print("⚠️ PDF 로더가 없어서 Baseline 문서를 로드할 수 없습니다.")
    
    if not baseline_docs:
        print("❌ Baseline 문서를 로드하지 못했습니다. RAG가 Baseline 증거를 찾지 못할 수 있습니다.")
        if LANGCHAIN_AVAILABLE:
            split_docs = [Document(page_content="No official baseline documents loaded.", metadata={"source": "N/A", "document_type": "baseline", "page": 0})]
        else:
            split_docs = [{"page_content": "No official baseline documents loaded.", "metadata": {"source": "N/A", "document_type": "baseline", "page": 0}}]
    else:
        # 텍스트 분할 (선택적)
        if TEXT_SPLITTER_AVAILABLE and LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            split_docs = text_splitter.split_documents(baseline_docs)
        else:
            # 간단한 텍스트 분할
            split_docs = []
            for doc in baseline_docs:
                content = doc.page_content
                chunks = [content[i:i+500] for i in range(0, len(content), 450)]
                for i, chunk in enumerate(chunks):
                    split_docs.append({
                        "page_content": chunk,
                        "metadata": {**doc.metadata, "chunk_id": i}
                    })
    
    # ChromaDB에 저장 (선택적)
    if CHROMA_AVAILABLE and embedding_model and LANGCHAIN_AVAILABLE:
        try:
            baseline_vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_model,
                persist_directory=baseline_embed_dir
            )
            print(f"✅ Baseline 메모리 구축 완료 ({len(split_docs)}개 청크)")
            return baseline_vectorstore
        except Exception as e:
            print(f"⚠️ ChromaDB 저장 실패: {e}")
            return None
    else:
        print("⚠️ ChromaDB가 없어서 Baseline 메모리를 파일로 저장합니다.")
        # JSON 파일로 저장
        baseline_data = []
        for doc in split_docs:
            if isinstance(doc, dict):
                baseline_data.append(doc)
            else:
                baseline_data.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        baseline_file = os.path.join(baseline_embed_dir, "baseline_data.json")
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Baseline 메모리 구축 완료 ({len(split_docs)}개 청크) - JSON 파일로 저장")
        return {"type": "json", "data": baseline_data}

print("✅ Baseline 메모리 함수 정의 완료")
# step5. 웹 크롤링 함수 정의 (Tavily 사용)
def crawl_web_content(keywords: List[str]) -> List[Dict[str, Any]]:
    """Tavily를 사용하여 웹에서 최신 AI 윤리 이슈 관련 기사 크롤링"""
    crawled_data = []
    
    if TAVILY_AVAILABLE:
        try:
            tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
            search_queries = []
            for keyword in keywords:
                search_queries.extend([
                    f"AI {keyword} 윤리 이슈",
                    f"AI {keyword} 편향성 문제",
                    f"AI {keyword} 개인정보보호",
                ])
            unique_queries = list(set(search_queries))[:5] # 최대 5개의 고유 쿼리로 제한
            
            for query in unique_queries:
                print(f"     - Tavily 검색 중: {query}...")
                try:
                    results = tavily.search(
                        query=query, 
                        search_depth="advanced", 
                        max_results=5, 
                        include_raw_content=True
                    )
                    for result in results.get("results", []):
                        if result.get("content"):
                            crawled_data.append({
                                "title": result.get("title", "No Title"),
                                "content": result["content"],
                                "source": result.get("url", "Unknown Source"),
                                "url": result.get("url", ""),
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "category": "issue"
                            })
                except Exception as e:
                    print(f"⚠️ Tavily 검색 실패 ({query}): {e}")
                    continue
        except Exception as e:
            print(f"⚠️ Tavily 초기화 실패: {e}")
    else:
        print("⚠️ Tavily가 없어서 웹 크롤링을 건너뜁니다.")
        # 더미 데이터 생성
        for keyword in keywords:
            crawled_data.append({
                "title": f"AI {keyword} 윤리 이슈 관련 기사",
                "content": f"최근 AI {keyword} 관련 윤리적 문제가 사회적 논란이 되고 있습니다. 이는 AI 시스템의 공정성과 투명성에 대한 우려를 나타냅니다.",
                "source": "AI Ethics Today",
                "url": f"https://example.com/ai-{keyword}-ethics",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "category": "issue"
            })

    # 필터링 로직 (300자 이상, 선정적 표현 제거, 중복 URL 제거)
    filtered_data = []
    seen_urls = set()
    for item in crawled_data:
        if len(item["content"]) >= 300:
            if not any(word in item["content"].lower() for word in ["충격", "폭로", "clickbait", "논란의", "대박"]):
                if item["url"] not in seen_urls:
                    filtered_data.append(item)
                    seen_urls.add(item["url"])

    print(f"✅ 웹 크롤링 완료 ({len(filtered_data)}개 문서)")
    return filtered_data

print("✅ 웹 크롤링 함수 정의 완료")

# step6. Issue 메모리 구축 (웹 크롤링 결과를 RAG에 저장)
def build_issue_memory(keywords: List[str]):
    """웹 크롤링 결과 기반 Issue 메모리 (Vectorstore) 구축"""
    crawled_data = crawl_web_content(keywords)
    issue_docs = []
    
    for item in crawled_data:
        if LANGCHAIN_AVAILABLE:
            doc = Document(
                page_content=f"[이슈: {item['category']}] {item['title']}\n\n{item['content']}",
                metadata={
                    "document_type": "issue",
                    "source": item["source"],
                    "url": item["url"],
                    "date": item["date"],
                    "category": item["category"], # 임시로 'issue'로 설정
                    "title": item["title"]
                }
            )
        else:
            doc = {
                "page_content": f"[이슈: {item['category']}] {item['title']}\n\n{item['content']}",
                "metadata": {
                    "document_type": "issue",
                    "source": item["source"],
                    "url": item["url"],
                    "date": item["date"],
                    "category": item["category"],
                    "title": item["title"]
                }
            }
        issue_docs.append(doc)
    
    if issue_docs:
        # Issue 문서는 ChromaDB에 저장 (선택적)
        if CHROMA_AVAILABLE and embedding_model and LANGCHAIN_AVAILABLE:
            try:
                issue_vectorstore = Chroma.from_documents(
                    documents=issue_docs,
                    embedding=embedding_model,
                    persist_directory=issue_embed_dir
                )
                print(f"✅ Issue 메모리 구축 완료 ({len(issue_docs)}개 문서)")
                return issue_vectorstore
            except Exception as e:
                print(f"⚠️ ChromaDB 저장 실패: {e}")
                return None
        else:
            # JSON 파일로 저장
            issue_data = []
            for doc in issue_docs:
                if isinstance(doc, dict):
                    issue_data.append(doc)
                else:
                    issue_data.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            issue_file = os.path.join(issue_embed_dir, "issue_data.json")
            with open(issue_file, 'w', encoding='utf-8') as f:
                json.dump(issue_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Issue 메모리 구축 완료 ({len(issue_docs)}개 문서) - JSON 파일로 저장")
            return {"type": "json", "data": issue_data}
    else:
        print("⚠️ 크롤링된 데이터가 없습니다.")
        return None

print("✅ Issue 메모리 구축 함수 정의 완료")


# 💡 신규 함수 정의: LLM을 이용한 증거 요약
def summarize_evidence_with_llm(docs: List, query: str) -> List[Dict[str, Any]]:
    """검색된 Document 목록을 LLM을 사용하여 요약하고 세부 정보와 결합합니다."""
    if not llm:
        print("⚠️ LLM이 초기화되지 않아 요약을 건너뜁니다.")
        # LLM 없이 간단한 요약 생성
        summarized_results = []
        for doc in docs:
            if isinstance(doc, dict):
                content = doc["page_content"]
                metadata = doc["metadata"]
            else:
                content = doc.page_content
                metadata = doc.metadata
            
            source = metadata.get("source", metadata.get("url", "Unknown"))
            doc_type = metadata.get("document_type", "Unknown")
            category = metadata.get("category", "N/A")

            # 문서 타입에 따른 청크 정보 설정
            if doc_type == "baseline":
                chunk_info = f"(페이지 {metadata.get('page', 'N/A')}의 내용)"
                score = 0.8 # Baseline 가중치
            else: # issue
                chunk_info = "(웹 기사 원문)"
                score = 0.2 # Issue 가중치

            # 간단한 요약 생성
            summary = f"{source}에서 {category} 관련 정보를 찾았습니다. {content[:100]}..."

            summarized_results.append({
                "category": category,
                "document_type": doc_type,
                "source": source,
                "chunk_info": chunk_info,
                "score": score,
                "summary": summary, 
                "content_excerpt": content[:300] + "...",
                "full_content": content
            })
        
        return summarized_results

    summarized_results = []
    
    if LANGCHAIN_AVAILABLE:
        summary_prompt_template = """당신은 AI 윤리 리스크 진단 전문가입니다. 다음 정보를 분석하여 한국어로 3줄 이내의 간결하고 핵심적인 요약을 제공하세요.
        이 요약은 'AI 서비스 {query}의 윤리 리스크'에 대한 근거로 사용될 것입니다.
        ---
        문서 출처: {source} ({document_type}) {chunk_info}
        문서 내용: {content}
        ---
        요약:"""
        summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["query", "source", "document_type", "content", "chunk_info"])

    for doc in docs:
        if isinstance(doc, dict):
            content = doc["page_content"]
            metadata = doc["metadata"]
        else:
            content = doc.page_content
            metadata = doc.metadata
        
        source = metadata.get("source", metadata.get("url", "Unknown"))
        doc_type = metadata.get("document_type", "Unknown")
        category = metadata.get("category", "N/A")

        # 문서 타입에 따른 청크 정보 설정
        if doc_type == "baseline":
            chunk_info = f"(페이지 {metadata.get('page', 'N/A')}의 내용)"
            score = 0.8 # Baseline 가중치
        else: # issue
            chunk_info = "(웹 기사 원문)"
            score = 0.2 # Issue 가중치

        if LANGCHAIN_AVAILABLE:
            # 프롬프트 구성 및 요약 생성
            prompt_value = summary_prompt.invoke({
                "query": query,
                "source": source,
                "document_type": doc_type,
                "content": content,
                "chunk_info": chunk_info
            })
            
            try:
                # LLM 호출
                summary_response = llm.invoke(prompt_value.to_string())
                summary = summary_response.content.strip()
            except Exception as e:
                summary = f"LLM 요약 실패. Error: {e}"
        else:
            # 간단한 요약 생성
            summary = f"{source}에서 {category} 관련 정보를 찾았습니다. {content[:100]}..."
        
        # Risk Assessor 에이전트에 전달할 상세 구조
        summarized_results.append({
            "category": category,
            "document_type": doc_type,
            "source": source,
            "chunk_info": chunk_info, # PDF 페이지 또는 웹 기사 여부
            "score": score,
            "summary": summary, 
            "content_excerpt": content[:300] + "...", # 원문 내용의 일부 (너무 길어지지 않도록)
            "full_content": content # Risk Assessor에서 필요할 경우를 대비하여 전체 원문도 전달
        })
        
    return summarized_results

# step7. 증거 수집 함수 정의 (가중치 8:2 적용)
def collect_evidence(service_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    서비스 프로파일 기반 증거 수집 (Baseline 0.8 : Issue 0.2)
    - Risk Assessor에게 전달할 증거 소스 목록 및 가중치 점수, 요약 포함
    """
    
    service_name = service_profile.get("service_name", "")
    # 💡 Service Profiler에서 변경된 키를 사용하도록 수정
    risk_categories = service_profile.get("diagnosed_risk_categories", []) 
    service_type = service_profile.get("service_type", "")
    
    print(f"\n🔍 증거 수집 시작: {service_name}")
    
    # 메모리 구축
    baseline_vectorstore = build_baseline_memory()
    issue_vectorstore = build_issue_memory(risk_categories)
    
    evidence_results = {
        "query": service_name,
        "weights": {"baseline": 0.8, "issue": 0.2},
        "scores": {},
        "baseline_sources": [],
        "issue_sources": []
    }
    
    all_docs_to_summarize = []
    
    # 1. 각 리스크 카테고리별 증거 검색
    for category in risk_categories:
        
        # Baseline 검색 쿼리 강화
        baseline_query = f"{service_name} {category} 리스크 {service_type} (EU AI Act, OECD, UNESCO 윤리 기준)"
        issue_query = f"최신 뉴스 논문 AI {service_name} {category} 문제"
        
        print(f"\n     📊 {category.upper()} 리스크 검색 중...")
        
        # Baseline 검색 (선택적)
        baseline_docs = []
        if baseline_vectorstore and hasattr(baseline_vectorstore, 'similarity_search'):
            try:
                baseline_docs = baseline_vectorstore.similarity_search(baseline_query, k=3)
            except Exception as e:
                print(f"⚠️ Baseline 검색 실패: {e}")
                baseline_docs = []
        elif isinstance(baseline_vectorstore, dict) and baseline_vectorstore.get("type") == "json":
            # JSON 데이터에서 간단한 검색
            baseline_data = baseline_vectorstore["data"]
            for doc in baseline_data[:3]:  # 최대 3개
                if category.lower() in doc["page_content"].lower() or "baseline" in doc["metadata"].get("document_type", ""):
                    baseline_docs.append(doc)
        
        # Issue 검색 (선택적)
        issue_docs = []
        if issue_vectorstore and hasattr(issue_vectorstore, 'similarity_search'):
            try:
                issue_docs = issue_vectorstore.similarity_search(issue_query, k=2)
            except Exception as e:
                print(f"⚠️ Issue 검색 실패: {e}")
                issue_docs = []
        elif isinstance(issue_vectorstore, dict) and issue_vectorstore.get("type") == "json":
            # JSON 데이터에서 간단한 검색
            issue_data = issue_vectorstore["data"]
            for doc in issue_data[:2]:  # 최대 2개
                if category.lower() in doc["page_content"].lower() or "issue" in doc["metadata"].get("document_type", ""):
                    issue_docs.append(doc)
            
        # 검색된 문서를 요약 대상 리스트에 추가 (카테고리 메타데이터 부여)
        for doc in baseline_docs:
            if isinstance(doc, dict):
                doc["metadata"] = doc.get("metadata", {})
                doc["metadata"]['category'] = category
            else:
                doc.metadata['category'] = category
            all_docs_to_summarize.append(doc)
            
        for doc in issue_docs:
            if isinstance(doc, dict):
                doc["metadata"] = doc.get("metadata", {})
                doc["metadata"]['category'] = category
            else:
                doc.metadata['category'] = category
            all_docs_to_summarize.append(doc)
            
        # 종합 점수 계산 (참고용)
        baseline_weight = 0.8
        issue_weight = 0.2 if issue_docs else 0.0
        total_score = (len(baseline_docs) > 0) * baseline_weight + (len(issue_docs) > 0) * issue_weight
        evidence_results["scores"][category] = total_score
        
        print(f" - 검색된 Baseline 청크: {len(baseline_docs)}개")
        print(f" - 검색된 Issue 문서: {len(issue_docs)}개")

    print("\n📝 검색된 증거들을 LLM을 사용하여 요약 중...")
    
    # 2. 통합 요약 및 데이터 구조화
    summarized_evidences = summarize_evidence_with_llm(all_docs_to_summarize, service_name)
    
    # 3. 최종 결과 리스트에 추가
    for evidence in summarized_evidences:
        if evidence['document_type'] == 'baseline':
            evidence_results["baseline_sources"].append(evidence)
        elif evidence['document_type'] == 'issue':
            evidence_results["issue_sources"].append(evidence)
    
    print(f"\n✅ 증거 수집 및 요약 완료!")
    return evidence_results

print("✅ 증거 수집 함수 정의 완료")

# step8. State 기반 실행 함수 정의
import sys
sys.path.append('..')
from state_manager import load_state, save_state, update_status

def evidence_collector_execute():
    """Evidence Collector 실행 함수"""
    print("\n" + "="*60)
    print("🔍 Evidence Collector 시작...")
    print("="*60)
    
    # State 로드
    state = load_state()
    
    # Service Profiler 결과 확인
    if state.get("status", {}).get("service_profiler") != "completed":
        print("❌ Service Profiler가 완료되지 않았습니다.")
        return state
    
    service_profile = state.get("service_profile", {})
    if not service_profile:
        print("❌ Service Profile이 없습니다.")
        return state
    
    # 증거 수집 실행
    evidence_result = collect_evidence(service_profile)
    state["evidence_data"] = evidence_result
    
    # State 저장
    save_state(state)
    update_status(state, "evidence_collector", "completed")
    
    print(f"✅ 증거 수집 완료 - {len(evidence_result['baseline_sources'])}개 Baseline + {len(evidence_result['issue_sources'])}개 Issue")
    
    return state

print("✅ Evidence Collector 실행 함수 정의 완료")

# 테스트 코드는 주석 처리됨 - 실제 실행 시에는 evidence_collector_execute() 함수만 호출