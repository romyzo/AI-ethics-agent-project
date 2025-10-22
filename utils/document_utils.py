"""
Document Processing Utility Functions

문서 로딩, 청킹, 메타데이터 추출 등 문서 처리 관련 함수들
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# 문서 처리 라이브러리
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)

from .logging_utils import log_agent_activity, log_error


def load_document(file_path: str) -> str:
    """
    다양한 형식의 문서를 텍스트로 로드
    
    지원 형식: PDF, DOCX, PPTX, TXT, MD
    
    Args:
        file_path: 문서 파일 경로
        
    Returns:
        추출된 텍스트
        
    Raises:
        ValueError: 지원하지 않는 파일 형식
        FileNotFoundError: 파일이 존재하지 않음
        
    Examples:
        >>> text = load_document("service_description.pdf")
        >>> print(f"Loaded {len(text)} characters")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    log_agent_activity(
        agent_name="DocumentLoader",
        action="load_start",
        data={"file": str(file_path), "size_bytes": file_path.stat().st_size}
    )
    
    try:
        # 파일 확장자에 따라 적절한 로더 선택
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif suffix == '.docx':
            loader = Docx2txtLoader(str(file_path))
        elif suffix == '.pptx':
            loader = UnstructuredPowerPointLoader(str(file_path))
        elif suffix in ['.txt', '.md']:
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # 문서 로드
        documents = loader.load()
        
        # 모든 페이지/섹션의 텍스트를 하나로 결합
        text = "\n\n".join([doc.page_content for doc in documents])
        
        log_agent_activity(
            agent_name="DocumentLoader",
            action="load_complete",
            data={
                "file": str(file_path),
                "text_length": len(text),
                "num_pages": len(documents)
            }
        )
        
        return text
    
    except Exception as e:
        log_error("DocumentLoader", e, {"file": str(file_path)})
        raise


def load_multiple_documents(file_paths: List[str]) -> Dict[str, str]:
    """
    여러 문서를 동시에 로드
    
    Args:
        file_paths: 파일 경로 리스트
        
    Returns:
        {파일명: 텍스트} 딕셔너리
        
    Examples:
        >>> files = ["doc1.pdf", "doc2.docx"]
        >>> documents = load_multiple_documents(files)
        >>> for filename, text in documents.items():
        ...     print(f"{filename}: {len(text)} chars")
    """
    results = {}
    
    for file_path in file_paths:
        try:
            text = load_document(file_path)
            results[Path(file_path).name] = text
        except Exception as e:
            log_error("DocumentLoader", e, {"file": file_path})
            # 실패한 파일은 건너뛰고 계속 진행
            results[Path(file_path).name] = ""
    
    return results


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    긴 텍스트를 청크로 분할
    
    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기 (문자 수)
        chunk_overlap: 청크 간 겹치는 부분 (문자 수)
        separators: 우선 분할 기준 (기본값: 문단, 문장, 단어)
        
    Returns:
        청크 리스트
        
    Examples:
        >>> long_text = "..." * 10000
        >>> chunks = chunk_text(long_text, chunk_size=500)
        >>> print(f"Split into {len(chunks)} chunks")
    """
    if separators is None:
        # 기본 분리자: 문단 -> 문장 -> 단어
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    
    log_agent_activity(
        agent_name="DocumentProcessor",
        action="text_chunking",
        data={
            "original_length": len(text),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": chunk_overlap
        }
    )
    
    return chunks


def extract_metadata(document: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    문서에서 메타데이터 추출
    
    Args:
        document: 문서 텍스트
        file_path: 파일 경로 (선택사항)
        
    Returns:
        메타데이터 딕셔너리
        
    Examples:
        >>> text = load_document("report.pdf")
        >>> metadata = extract_metadata(text, "report.pdf")
        >>> print(metadata["word_count"])
    """
    metadata = {
        "character_count": len(document),
        "word_count": len(document.split()),
        "line_count": len(document.split('\n')),
    }
    
    if file_path:
        path = Path(file_path)
        metadata.update({
            "filename": path.name,
            "file_extension": path.suffix,
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
        })
    
    # 간단한 내용 분석
    metadata.update({
        "has_urls": bool(re.search(r'https?://', document)),
        "has_emails": bool(re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', document)),
        "paragraph_count": len([p for p in document.split('\n\n') if p.strip()]),
    })
    
    # AI/윤리 관련 키워드 탐지
    ethics_keywords = [
        'bias', 'fairness', 'discrimination', 'privacy', 'transparency',
        'accountability', 'ethics', 'responsible', 'safety', 'security',
        '편향', '공정', '차별', '개인정보', '투명성', '책임', '윤리', '안전', '보안'
    ]
    
    keyword_counts = {}
    for keyword in ethics_keywords:
        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', document, re.IGNORECASE))
        if count > 0:
            keyword_counts[keyword] = count
    
    metadata["ethics_keywords"] = keyword_counts
    
    return metadata


def clean_text(text: str) -> str:
    """
    텍스트 정제 (불필요한 공백, 특수문자 제거)
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정제된 텍스트
        
    Examples:
        >>> dirty_text = "Hello   World\\n\\n\\n\\nTest"
        >>> clean = clean_text(dirty_text)
        >>> print(clean)  # "Hello World\n\nTest"
    """
    # 연속된 공백을 하나로
    text = re.sub(r' +', ' ', text)
    
    # 3개 이상의 연속 줄바꿈을 2개로
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def extract_sections(text: str) -> Dict[str, str]:
    """
    문서를 섹션별로 분리
    
    제목을 기준으로 섹션을 나눔
    
    Args:
        text: 문서 텍스트
        
    Returns:
        {섹션명: 내용} 딕셔너리
        
    Examples:
        >>> text = "# Introduction\\nThis is intro.\\n\\n# Methods\\nThis is methods."
        >>> sections = extract_sections(text)
        >>> print(sections.keys())
    """
    sections = {}
    
    # 마크다운 스타일 헤더 찾기 (# Title)
    md_pattern = r'^#{1,3}\s+(.+)$'
    
    lines = text.split('\n')
    current_section = "Introduction"
    current_content = []
    
    for line in lines:
        md_match = re.match(md_pattern, line)
        
        if md_match:
            # 이전 섹션 저장
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # 새 섹션 시작
            current_section = md_match.group(1).strip()
            current_content = []
        else:
            current_content.append(line)
    
    # 마지막 섹션 저장
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def summarize_document(text: str, max_length: int = 500) -> str:
    """
    문서의 간단한 요약 생성 (처음 N 문자)
    
    Args:
        text: 문서 텍스트
        max_length: 최대 길이
        
    Returns:
        요약 텍스트
        
    Examples:
        >>> long_text = "..." * 1000
        >>> summary = summarize_document(long_text, max_length=200)
        >>> print(len(summary) <= 200)  # True
    """
    if len(text) <= max_length:
        return text
    
    # 문장 경계에서 자르기
    truncated = text[:max_length]
    
    # 마지막 완전한 문장까지만
    last_period = truncated.rfind('.')
    if last_period > max_length * 0.7:  # 최소 70% 이상 유지
        truncated = truncated[:last_period + 1]
    
    return truncated + "..."


def count_tokens_approx(text: str) -> int:
    """
    텍스트의 대략적인 토큰 수 추정
    
    정확한 계산은 llm_utils.calculate_token_count() 사용
    
    Args:
        text: 텍스트
        
    Returns:
        추정 토큰 수
    """
    # 간단한 추정: 1 토큰 ≈ 4 문자 (영어 기준)
    # 한글은 약간 더 효율적
    char_count = len(text)
    word_count = len(text.split())
    
    # 영어와 한글 혼합 고려
    estimated_tokens = (char_count / 4 + word_count) / 2
    
    return int(estimated_tokens)


def is_document_too_large(text: str, max_tokens: int = 100000) -> bool:
    """
    문서가 너무 큰지 확인
    
    Args:
        text: 문서 텍스트
        max_tokens: 최대 토큰 수
        
    Returns:
        너무 크면 True
    """
    estimated = count_tokens_approx(text)
    return estimated > max_tokens


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 샘플 텍스트
    sample_text = """
    # AI Ethics Assessment
    
    This document describes our AI service.
    
    ## Privacy Considerations
    
    We handle user data with care. Privacy is important.
    
    ## Bias Mitigation
    
    We test for bias regularly.
    """
    
    # 메타데이터 추출 테스트
    metadata = extract_metadata(sample_text)
    print("Metadata:", metadata)
    
    # 섹션 추출 테스트
    sections = extract_sections(sample_text)
    print("Sections:", list(sections.keys()))
    
    # 청킹 테스트
    chunks = chunk_text(sample_text, chunk_size=100)
    print(f"Chunks: {len(chunks)}")

