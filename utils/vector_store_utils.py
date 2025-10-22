"""
Vector Store Utility Functions

벡터 스토어 초기화, 문서 추가, 유사도 검색 등 RAG 관련 함수들
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Vector Store
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .logging_utils import log_agent_activity, log_error
from .document_utils import chunk_text

# 환경 변수
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "ai_ethics_knowledge_base")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")


class VectorStoreManager:
    """벡터 스토어 관리자"""
    
    def __init__(
        self, 
        store_type: str = "chroma",
        embedding_model: Optional[str] = None
    ):
        """
        Args:
            store_type: 벡터 스토어 타입 ("chroma" 또는 "faiss")
            embedding_model: 임베딩 모델명
        """
        self.store_type = store_type.lower()
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", 
            "text-embedding-3-large"
        )
        
        # 임베딩 객체 생성
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.vector_store = None
        
        log_agent_activity(
            agent_name="VectorStore",
            action="manager_initialized",
            data={
                "store_type": self.store_type,
                "embedding_model": self.embedding_model
            }
        )
    
    def initialize_store(self, load_existing: bool = True) -> Any:
        """
        벡터 스토어 초기화 또는 로드
        
        Args:
            load_existing: 기존 스토어 로드 여부
            
        Returns:
            벡터 스토어 객체
        """
        try:
            if self.store_type == "chroma":
                persist_dir = Path(CHROMA_PERSIST_DIR)
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                if load_existing and (persist_dir / "chroma.sqlite3").exists():
                    # 기존 스토어 로드
                    self.vector_store = Chroma(
                        collection_name=CHROMA_COLLECTION,
                        embedding_function=self.embeddings,
                        persist_directory=str(persist_dir)
                    )
                    log_agent_activity(
                        agent_name="VectorStore",
                        action="loaded_existing_store",
                        data={"type": "chroma", "path": str(persist_dir)}
                    )
                else:
                    # 새 스토어 생성
                    self.vector_store = Chroma(
                        collection_name=CHROMA_COLLECTION,
                        embedding_function=self.embeddings,
                        persist_directory=str(persist_dir)
                    )
                    log_agent_activity(
                        agent_name="VectorStore",
                        action="created_new_store",
                        data={"type": "chroma", "path": str(persist_dir)}
                    )
            
            elif self.store_type == "faiss":
                index_path = Path(FAISS_INDEX_PATH)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                
                if load_existing and index_path.exists():
                    # 기존 FAISS 인덱스 로드
                    self.vector_store = FAISS.load_local(
                        str(index_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    log_agent_activity(
                        agent_name="VectorStore",
                        action="loaded_existing_store",
                        data={"type": "faiss", "path": str(index_path)}
                    )
                else:
                    # FAISS는 문서가 추가될 때 생성됨
                    self.vector_store = None
                    log_agent_activity(
                        agent_name="VectorStore",
                        action="faiss_ready",
                        data={"note": "Will be created when documents are added"}
                    )
            
            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")
            
            return self.vector_store
        
        except Exception as e:
            log_error("VectorStore", e, {"action": "initialize_store"})
            raise
    
    def add_documents(
        self, 
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> int:
        """
        문서를 벡터 스토어에 추가
        
        Args:
            texts: 문서 텍스트 리스트
            metadatas: 각 문서의 메타데이터 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            
        Returns:
            추가된 청크 개수
        """
        try:
            all_chunks = []
            all_metadatas = []
            
            # 각 문서를 청크로 분할
            for idx, text in enumerate(texts):
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                
                # 메타데이터 설정
                if metadatas and idx < len(metadatas):
                    base_metadata = metadatas[idx]
                else:
                    base_metadata = {"doc_id": idx}
                
                # 각 청크에 메타데이터 추가 (청크 번호 포함)
                for chunk_idx in range(len(chunks)):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_idx
                    chunk_metadata["total_chunks"] = len(chunks)
                    all_metadatas.append(chunk_metadata)
            
            # Document 객체 생성
            documents = [
                Document(page_content=chunk, metadata=meta)
                for chunk, meta in zip(all_chunks, all_metadatas)
            ]
            
            # 벡터 스토어에 추가
            if self.store_type == "chroma":
                if self.vector_store is None:
                    self.initialize_store(load_existing=False)
                
                self.vector_store.add_documents(documents)
                
            elif self.store_type == "faiss":
                if self.vector_store is None:
                    # FAISS 스토어 생성
                    self.vector_store = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
                else:
                    # 기존 스토어에 추가
                    self.vector_store.add_documents(documents)
                
                # FAISS 저장
                self.vector_store.save_local(FAISS_INDEX_PATH)
            
            log_agent_activity(
                agent_name="VectorStore",
                action="documents_added",
                data={
                    "num_documents": len(texts),
                    "num_chunks": len(all_chunks),
                    "store_type": self.store_type
                }
            )
            
            return len(all_chunks)
        
        except Exception as e:
            log_error("VectorStore", e, {"action": "add_documents"})
            raise
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 개수
            filter_metadata: 메타데이터 필터
            score_threshold: 유사도 임계값 (이상만 반환)
            
        Returns:
            (Document, 유사도 점수) 튜플 리스트
        """
        try:
            if self.vector_store is None:
                self.initialize_store(load_existing=True)
            
            if self.vector_store is None:
                log_agent_activity(
                    agent_name="VectorStore",
                    action="search_failed",
                    data={"reason": "No vector store available"}
                )
                return []
            
            # 유사도 검색 수행
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_metadata
            )
            
            # 임계값 필터링
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            log_agent_activity(
                agent_name="VectorStore",
                action="search_completed",
                data={
                    "query_length": len(query),
                    "top_k": top_k,
                    "results_found": len(results)
                }
            )
            
            return results
        
        except Exception as e:
            log_error("VectorStore", e, {"action": "search_similar", "query": query[:100]})
            return []
    
    def delete_collection(self):
        """벡터 스토어 컬렉션 삭제"""
        try:
            if self.store_type == "chroma":
                self.vector_store.delete_collection()
                log_agent_activity(
                    agent_name="VectorStore",
                    action="collection_deleted",
                    data={"type": "chroma"}
                )
            elif self.store_type == "faiss":
                index_path = Path(FAISS_INDEX_PATH)
                if index_path.exists():
                    import shutil
                    shutil.rmtree(index_path)
                log_agent_activity(
                    agent_name="VectorStore",
                    action="collection_deleted",
                    data={"type": "faiss"}
                )
        except Exception as e:
            log_error("VectorStore", e, {"action": "delete_collection"})


# 전역 매니저 인스턴스
_vector_store_manager = None

def get_vector_store_manager(
    store_type: str = "chroma",
    embedding_model: Optional[str] = None
) -> VectorStoreManager:
    """벡터 스토어 매니저 싱글톤 반환"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager(store_type, embedding_model)
    return _vector_store_manager


def initialize_vector_store(
    store_type: str = "chroma",
    load_existing: bool = True
) -> Any:
    """
    벡터 스토어 초기화
    
    Args:
        store_type: "chroma" 또는 "faiss"
        load_existing: 기존 스토어 로드 여부
        
    Returns:
        벡터 스토어 객체
        
    Examples:
        >>> store = initialize_vector_store("chroma")
        >>> print("Vector store initialized")
    """
    manager = get_vector_store_manager(store_type)
    return manager.initialize_store(load_existing)


def add_documents_to_store(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    store_type: str = "chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> int:
    """
    벡터 스토어에 문서 추가
    
    Args:
        texts: 문서 텍스트 리스트
        metadatas: 메타데이터 리스트
        store_type: 벡터 스토어 타입
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        
    Returns:
        추가된 청크 개수
        
    Examples:
        >>> texts = ["Document 1 content", "Document 2 content"]
        >>> count = add_documents_to_store(texts)
        >>> print(f"Added {count} chunks")
    """
    manager = get_vector_store_manager(store_type)
    return manager.add_documents(texts, metadatas, chunk_size, chunk_overlap)


def search_similar_documents(
    query: str,
    top_k: int = 5,
    store_type: str = "chroma",
    filter_metadata: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    유사한 문서 검색
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 개수
        store_type: 벡터 스토어 타입
        filter_metadata: 메타데이터 필터
        score_threshold: 유사도 임계값
        
    Returns:
        검색 결과 리스트 (각 항목은 content, metadata, score 포함)
        
    Examples:
        >>> results = search_similar_documents("AI bias issues", top_k=3)
        >>> for result in results:
        ...     print(f"Score: {result['score']}, Content: {result['content'][:100]}")
    """
    manager = get_vector_store_manager(store_type)
    results = manager.search_similar(query, top_k, filter_metadata, score_threshold)
    
    # 결과를 딕셔너리 형태로 변환
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })
    
    return formatted_results


def get_collection_stats(store_type: str = "chroma") -> Dict[str, Any]:
    """
    벡터 스토어 통계 정보 반환
    
    Args:
        store_type: 벡터 스토어 타입
        
    Returns:
        통계 정보 딕셔너리
    """
    try:
        manager = get_vector_store_manager(store_type)
        
        if manager.vector_store is None:
            manager.initialize_store(load_existing=True)
        
        if manager.vector_store is None:
            return {"status": "empty", "document_count": 0}
        
        # Chroma의 경우
        if store_type == "chroma":
            collection = manager.vector_store._collection
            count = collection.count()
            
            return {
                "status": "active",
                "document_count": count,
                "collection_name": CHROMA_COLLECTION,
                "persist_directory": CHROMA_PERSIST_DIR
            }
        
        # FAISS의 경우
        elif store_type == "faiss":
            index = manager.vector_store.index
            count = index.ntotal
            
            return {
                "status": "active",
                "document_count": count,
                "index_path": FAISS_INDEX_PATH
            }
    
    except Exception as e:
        log_error("VectorStore", e, {"action": "get_collection_stats"})
        return {"status": "error", "error": str(e)}


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 샘플
    print("Testing Vector Store Utils...")
    
    # 벡터 스토어 초기화
    store = initialize_vector_store("chroma", load_existing=False)
    print("✓ Vector store initialized")
    
    # 문서 추가
    sample_docs = [
        "AI ethics requires careful consideration of bias and fairness.",
        "Privacy protection is crucial in AI systems.",
        "Transparency and explainability help build trust in AI."
    ]
    
    count = add_documents_to_store(sample_docs)
    print(f"✓ Added {count} chunks")
    
    # 검색 테스트
    results = search_similar_documents("bias in AI", top_k=2)
    print(f"✓ Found {len(results)} similar documents")
    
    # 통계
    stats = get_collection_stats()
    print(f"✓ Collection stats: {stats}")

