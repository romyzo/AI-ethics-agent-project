# 🛡️ AI 윤리성 리스크 진단 에이전트 (v1.3)

본 프로젝트는 **EU AI Act**, **OECD AI Principles**, **UNESCO AI Ethics Recommendation** 기준을 통합하여  
AI 서비스의 윤리적 리스크(편향성, 프라이버시, 투명성, 안전성/거버넌스 등)를 자동으로 진단하고  
개선 권고안을 포함한 보고서를 생성하는 **LangGraph 기반 Agentic RAG 프로젝트**입니다.

---

## 🧭 Overview

- **Objective :**  
  최근 기업들은 경쟁적으로 AI 서비스를 상용화하지만,  
  사전 **윤리성·법적 리스크 점검 체계(Pre-deployment Check)** 는 여전히 미흡한 실정입니다.  
  본 프로젝트는 **AI 서비스의 윤리 리스크를 자동 평가하고 개선 권고안을 제시하는 시스템**을 구축하는 것을 목표로 합니다.

- **Methods :**  
  LangGraph 기반 Multi-Agent Workflow + Agentic RAG + Structured Markdown → PDF 자동화  

- **Tools :**  
  LangChain, LangGraph, OpenAI API, HuggingFace Embedding, Chroma/FAISS, PyMuPDF, Tavily Search

---

## ✨ Features

1️⃣ **다중 프레임 통합 윤리 진단**  
EU AI Act(법적 관점) + OECD(투명성·책임성) + UNESCO(포용성·인간중심 윤리) 기준을 통합하여  
**법적·도덕적·사회적 관점에서 다층 리스크 자동 분석**

2️⃣ **RAG 기반 최신 근거 결합**  
- 로컬 문서(EU, OECD, UNESCO PDF) + 웹 최신 뉴스/논문 (Tavily API) 하이브리드 검색  
- Chroma/FAISS 기반 벡터 임베딩으로 **근거 신뢰도 및 최신성 강화**

3️⃣ **LLM 기반 보고서 자동화**  
- 에이전트 결과를 통합하여 **Markdown → PDF 보고서 자동 생성**  
- 구조화된 보고서 섹션 : Executive Summary → Profile → Risk → Mitigation → Evidence → Conclusion

---

## 🧑‍🏫 교수님 피드백 정리

| 피드백 항목 | 반영 내용 | 학습 및 개선 효과 |
|:--|:--|:--|
| **1. 임베딩 단계의 경제성 확보 필요** | HuggingFace 임베딩 모델 적용 (`all-MiniLM-L6-v2`) | 임베딩 비용 0원, CPU 환경에서도 실행 가능 |
| **2. 임베딩 비용 구조 이해 필요** | 토큰 단위 과금 구조 및 chunking 과정 학습 | RAG 설계 시 비용 발생 지점 명확히 파악 |
| **3. 웹 크롤링 결과 재활용 필요** | Tavily 결과를 Vector DB에 저장 | 재실행 시 동일 근거 재활용, 일관성 향상 |
| **4. Baseline/Issue 분리 설계** | Chroma DB를 이중 저장소로 구성 | 검색 정확도 및 문맥 분리 강화 |
| **5. 보고서 자동화 완성도 강화** | Markdown → PDF 자동 변환 구조 확립 | 최종 결과물 제출용 수준 향상 |


## ✨ 수정 및 반영 내용 (v1.4 현재)

이 피드백을 반영하여 다음과 같이 개선했습니다:

| 구분 | 변경 전 | 변경 후 (현재 구조) |
|:--|:--|:--|
| **RAG 메모리 구조** | 웹 크롤링 결과만 단기 활용 | Baseline(공식문서) + Issue(웹 이슈) 2중 Chroma VectorDB 저장 |
| **임베딩 모델** | OpenAI Embedding (비용 부담) | HuggingFace `all-MiniLM-L6-v2` → CPU 기반, 무료/경량 실행 |
| **저장 방식** | 단기 캐싱(JSON 메모리) | Chroma 영구저장 + JSON Fallback 구조로 재실행 시 재활용 가능 |
| **검색 쿼리 전략** | 고정형 단일 질의 | 서비스명 + 리스크 카테고리 기반 동적 쿼리 생성 |
| **RAG 결과 활용** | LLM 프롬프트 입력에 직접 삽입 | 요약 + 근거 점수(weight)로 Risk Assessor 입력 구조 통일 |
| **결과 일관성** | 실행마다 변동 | State 기반 단계별 저장으로 동일 결과 재현 가능 |

이 구조 덕분에 이제 **한 번 수집된 웹/문서 근거를 여러 에이전트에서 재활용**할 수 있고,  
경제성(HuggingFace)과 품질(RAG 저장 재활용)을 모두 확보했습니다.

---

## 🧩 Tech Stack

| Category | Details |
|:--|:--|
| Framework | LangGraph, LangChain, Python |
| LLM | GPT-4o-mini via OpenAI API |
| Retrieval | Chroma, FAISS |
| Embedding | HuggingFace `all-MiniLM-L6-v2` |
| Document Loader | PyMuPDF (LangChain Loader) |
| Search | Tavily Search API |
| Report Export | pypandoc (wkhtmltopdf engine) |
| Optional Tools | Cohere Rerank, dotenv |

---

## 🧠 State Definition

| Key | Description |
|:--|:--|
| `service_description` | PDF 또는 입력 기반 서비스 설명 |
| `service_profile` | AI 서비스명, 유형, 리스크 카테고리 |
| `evidence_data` | EU/OECD/UNESCO 문서 + 웹 기사 기반 증거 |
| `assessment_result` | 카테고리별 리스크 등급 및 평가 요약 |
| `recommendation_result` | 각 리스크별 개선 권고안 |
| `final_report_markdown` | 최종 Markdown 보고서 원문 |
| `status` | 각 Agent 진행 상태 (pending / completed) |
| `report_pdf_path` | 생성된 PDF 파일 경로 |

---

## 🧱 Architecture
<img width="339" height="921" alt="image" src="https://github.com/user-attachments/assets/dec7c88d-c2fa-4a01-91e9-d58b14156256" />


## 📁 Directory Structure

ai_ethics_agent/
├── agents/
│   ├── service_profiler.ipynb / .py         # 서비스 분석 에이전트
│   ├── evidence_collector.ipynb / .py       # RAG 기반 증거 수집 에이전트
│   ├── risk_assessor.ipynb / .py            # 윤리 리스크 평가 에이전트
│   ├── mitigation_recommender.ipynb         # 개선 권고안 생성 에이전트
│   └── report_composer.ipynb / .py          # 최종 보고서 작성 에이전트
│
├── data/
│   ├── reference/                           # 참고 문서 (EU_AI_Act.pdf, OECD_Privacy_2024.pdf, UNESCO_Ethics_2021.pdf)
│   ├── crawled/                             # 웹 검색 데이터 저장
│   └── embeddings/                          # Chroma 벡터 DB 저장소
│
├── output/
│   └── reports/                             # 생성된 Markdown 및 PDF 보고서
│
├── utils/                                   # 공통 유틸 및 상태 관리 모듈
│   ├── state_manager.py                     # State 초기화 및 공유 로직
│   ├── agent_state.json                     # 각 에이전트 상태 저장 파일
│   └── __init__.py
│
├── main.py                                  # Supervisor 파이프라인 실행 스크립트
├── state.ipynb                              # LangGraph 상태 정의 노트북
├── .env                                     # API Key 및 환경 변수 설정
└── README.md



## 📚 Reference

- EU AI Act (2024)
- OECD AI Principles (2019)
- UNESCO AI Ethics Recommendation (2021)

- 
## 🧰 Troubleshooting & Update Log

| 날짜 | 수정 항목 | 개선 내용 |
|:--|:--|:--|
| 2025-10-22 | `.env` 인식 안 됨 → `load_dotenv()` 추가 | OpenAI / Tavily 키 로드 정상화 |
| 2025-10-22 | Baseline 문서 경로 오류 수정 | `data/reference` 기준 상대경로로 통일 |
| 2025-10-23 | pypandoc PDF 변환 오류 (`wkhtmltopdf` 엔진 누락) | 엔진 경로 지정 + CSS 스타일 적용 |
| 2025-10-23 | `state_manager` 저장 시 덮어쓰기 문제 | `updated_at` 필드 추가, 안전 저장 로직 적용 |
| 2025-10-23 | `EvidenceCollector`의 `metadata.category` 누락 | Risk Assessor 전달 데이터 구조 통일 |
| 2025-10-23 | JSON 직렬화 오류 (`ensure_ascii=False`) | LLM 반환 값 인코딩 문제 해결 |
| 2025-10-23 | `diagnosed_risk_categories` 키 불일치 | Profiler ↔ Collector key 매핑 수정 |
| 2025-10-23 | LangGraph 비주얼 누락 → Mermaid 추가 | Supervisor 전체 흐름 시각화 완료 |


## 🧩 기술 개념 정리 (학습 기록)

| 개념 | 학습 내용 요약 |
|:--|:--|
| **Embedding** | 텍스트를 벡터화하여 문서 간 의미적 유사도 비교에 활용. RAG의 핵심. |
| **Token 과금 구조** | OpenAI 모델은 입력/출력 모두 토큰 단위로 과금됨. Chunking할수록 비용 증가. |
| **RAG (Retrieval-Augmented Generation)** | LLM이 외부 문서(DB)에서 근거를 검색해 답변 품질을 높이는 구조. |
| **Chroma / FAISS** | 문서 벡터를 저장·검색하기 위한 Vector DB. Chroma는 persist(저장) 지원. |
| **LangGraph** | LLM Agent들을 순차/조건적으로 연결하는 워크플로우 프레임워크. |
| **pypandoc** | Markdown을 PDF로 변환하는 도구. 기업 보고서 수준의 서식 구현 가능. |

## 🧰 Git 사용법 학습 내용

| 기능 | 명령어 예시 | 이해 및 습득 내용 |
|:--|:--|:--|
| **저장소 복제** | `git clone <URL>` | GitHub 원격 저장소를 로컬 환경으로 복제 |
| **작업 브랜치 생성** | `git checkout -b dev` | main과 분리된 개발용 브랜치 운영 |
| **파일 추가 / 커밋** | `git add .` / `git commit -m "Add: new agent"` | 변경 내용을 단계별로 관리 |
| **원격 반영** | `git push origin dev` | 수정된 코드 푸시 및 팀 공유 |
| **상태 확인** | `git status` | 수정/추적 파일 상태를 실시간 확인 |
| **커밋 되돌리기** | `git reset --soft HEAD~1` | 실수한 커밋 취소 및 수정 가능 |
| **README 업데이트** | `git add README.md && git commit -m "Docs: Update README"` | 문서 변경도 버전별로 관리 |



## 👥 Contributors 
- **조혜림** : Prompt Engineering, Multi-Agent Design, RAG 구성, Architecture 설계
