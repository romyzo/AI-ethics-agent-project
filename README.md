# 🛡️ AI 윤리성 리스크 진단 에이전트
본 프로젝트는 **EU AI Act, OECD AI Principles, UNESCO AI Ethics** 기준을 통합하여  
AI 서비스의 윤리적 리스크(편향성, 프라이버시, 투명성, 안전성/거버넌스 등)를 자동으로 진단하고  
개선 권고안을 포함한 보고서를 생성하는 **LangGraph 기반 Agentic RAG 프로젝트**입니다.

## Overview

- Objective : 
  최근 기업들은 경쟁적으로 AI 서비스를 시장에 내놓고 있지만,  
  그 과정에서 **법적·윤리적 리스크에 대한 사전 점검 체계가 미흡**한 경우가 많습니다.  
  본 프로젝트는 이러한 현실적 문제를 해결하기 위해,  
  **AI 서비스 출시 전 단계(Pre-deployment)** 에서 윤리 리스크를 자동으로 진단하고  
  EU AI Act 기준의 위험 등급 및 개선 권고를 제시하는 **AI 윤리감사 에이전트 시스템**을 구축하는 것을 목표로 합니다.
- Methods : LangGraph 기반 Multi-Agent Workflow + Agentic RAG + Structured Markdown → PDF 자동화
- Tools : LangChain, LangGraph, OpenAI API, Chroma/FAISS, PyMuPDF, Tavily, Rerank

## Features

**1. 다중 프레임 통합 윤리 진단:**  
  EU AI Act, OECD, UNESCO 세 가지 기준을 조합하여 **법적 + 도덕적 + 사회적 리스크를 다층적으로 평가**
  › EU AI Act → 법적, 위험 기반 (정량 평가)
  › OECD → 투명성, 책임성 등 거버넌스 중심 (원칙 매핑)
  › UNESCO → 포용성, 인간 중심 윤리 기준 (사회적 리스크 인용)
  
**2. RAG 기반 근거 신뢰도 향상:**  
  로컬 KB(EU, OECD, UNESCO 문서) + 최신 뉴스·논문을 하이브리드 검색하여  
  **리스크 진단의 근거 정확도와 시의성을 강화**


## Tech Stack 

| Category   | Details                               |
|-------------|----------------------------------------|
| **Framework** | LangGraph, LangChain, Python |
| **LLM** | GPT-4o-mini via OpenAI API |
| **Retrieval** | FAISS, Chroma |
| **Embedding** | text-embedding-3-small (OpenAIEmbedding) |
| **PDF Export** | PyMuPDF (md2pdf) |
| **Optional Tools** | Tavily Search API, Cohere Rerank |


## Agents
 
| Agent | 역할 | 주요 기능 |
|--------|--------|-----------|
| **Supervisor Agent** | 전체 워크플로우 제어 | 각 노드의 실행 순서 및 상태 관리 |
| **Node A – Service Profiler** | 대상 서비스 분석 | 서비스 목적, 데이터 흐름, 사용영역 파악 |
| **Node B – Evidence Collector** | RAG 근거 수집 | 로컬 문서 + 웹 검색 기반 다층 근거 확보 |
| **Node C – RAG Indexer** | 임베딩 및 벡터스토어 구축 | Loader → Splitter → Embedding → VectorDB |
| **Node D – Risk Assessor** | 리스크 진단 및 등급 분류 | 항목별 점수 산출 및 위험 수준 평가 |
| **Node E – Mitigation Recommender** | 개선 권고안 생성 | 리스크별 개선 방향 및 근거 프레임 제시 |
| **Node F – Report Composer** | 보고서 자동 작성 | Markdown → PDF 변환 및 저장 |
| **Quality Gate** | 품질 관리 | 근거 부족 시 Evidence Collector 루프 재실행 |

## State 

| Key | Description |
|------|--------------|
| `services` | 입력된 AI 서비스 리스트 (최대 3개) |
| `profiles` | 각 서비스의 기능·사용 목적 요약 |
| `evidences` | RAG/웹검색으로 수집된 근거 문서 |
| `vector_ready` | 벡터스토어 구축 완료 여부 |
| `results` | 서비스별 리스크 점수·등급·요약 결과 |
| `report_md` | Markdown 보고서 원문 |
| `done` | 프로세스 완료 여부 (True/False) |


## Architecture
<img width="312" height="750" alt="image" src="https://github.com/user-attachments/assets/6b277ad8-4c65-4596-9edd-35184251f575" />


## Directory Structure

ai-ethics-agent/
├─ src/
│  ├─ main.py                # 실행 스크립트
│  ├─ graph.py               # LangGraph 흐름 정의
│  ├─ state.py               # 상태(State) 스키마
│  ├─ prompts.py             # 프롬프트 템플릿
│  ├─ retriever.py           # 검색 및 문서 로더
│  └─ nodes/                 # 개별 노드 모듈
│     ├─ profile.py          # Node A - 서비스 분석
│     ├─ collect.py          # Node B - 증거 수집
│     ├─ indexer.py          # Node C - 벡터 DB 구축
│     ├─ assess.py           # Node D - 리스크 평가
│     ├─ mitigate.py         # Node E - 개선 권고
│     └─ compose.py          # Node F - 보고서 생성
│
├─ kb/                       # 기준 문서 저장소 (EU, OECD, UNESCO)
│  ├─ eu_ai_act_summary.md
│  ├─ oecd_ai_principles.md
│  └─ unesco_ai_ethics.md
│
├─ vectorstore/              # 임베딩 DB (Chroma/FAISS)
├─ reports/                  # 결과 보고서 (Markdown/PDF)
├─ .env.example              # 환경 변수 예시 파일
├─ requirements.txt          # 의존성 리스트
└─ README.md                 # 프로젝트 설명 문서

## Reference

- EU AI Act (2024)
- OECD AI Principles (2019)
- UNESCO AI Ethics Recommendation (2021)
+ And more ! (Maybe..)


## Contributors 
- **조혜림** : Prompt Engineering, Multi-Agent Design, RAG 구성, Architecture 설계
