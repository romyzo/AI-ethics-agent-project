"""
Logging Utility Functions

로깅, 모니터링 관련 공통 함수들
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import time

# 로그 디렉토리 생성
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class AgentLogger:
    """에이전트 활동 로거"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.loggers = {}
        self.performance_data = {}
        
    def get_logger(self, name: str) -> logging.Logger:
        """
        지정된 이름의 로거 반환 (없으면 생성)
        
        Args:
            name: 로거 이름 (보통 에이전트 이름)
            
        Returns:
            Logger 인스턴스
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # 기존 핸들러 제거 (중복 방지)
        logger.handlers = []
        
        # 파일 핸들러 (JSON 형식)
        log_file = LOG_DIR / f"{name.lower().replace(' ', '_')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.loggers[name] = logger
        return logger


# 전역 로거 인스턴스
_agent_logger = None

def get_agent_logger() -> AgentLogger:
    """에이전트 로거 싱글톤 반환"""
    global _agent_logger
    if _agent_logger is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        _agent_logger = AgentLogger(log_level)
    return _agent_logger


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    로거 설정 및 반환
    
    Args:
        name: 로거 이름
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        설정된 Logger 인스턴스
        
    Examples:
        >>> logger = setup_logger("ServiceProfiler")
        >>> logger.info("Starting service analysis...")
    """
    agent_logger = get_agent_logger()
    return agent_logger.get_logger(name)


def log_agent_activity(
    agent_name: str,
    action: str,
    data: Optional[Dict[str, Any]] = None,
    level: str = "INFO"
):
    """
    에이전트 활동 로그 기록
    
    Args:
        agent_name: 에이전트 이름
        action: 수행한 작업
        data: 추가 데이터 (딕셔너리)
        level: 로그 레벨
        
    Examples:
        >>> log_agent_activity(
        ...     agent_name="Risk Assessor",
        ...     action="risk_evaluation_complete",
        ...     data={"risk_level": "high", "confidence": 0.95}
        ... )
    """
    logger = setup_logger(agent_name)
    
    # 로그 메시지 구성
    message = f"[{action}]"
    if data:
        message += f" {json.dumps(data, ensure_ascii=False)}"
    
    # 레벨에 따라 로깅
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
    
    # JSON 로그 파일에도 기록
    log_to_json_file(agent_name, action, data, level)


def log_performance_metrics(
    agent_name: str,
    duration: float,
    tokens_used: Optional[int] = None,
    memory_used: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """
    성능 메트릭 로그 기록
    
    Args:
        agent_name: 에이전트 이름
        duration: 실행 시간 (초)
        tokens_used: 사용된 토큰 수
        memory_used: 사용된 메모리 (MB)
        additional_metrics: 추가 메트릭
        
    Examples:
        >>> start_time = time.time()
        >>> # ... 작업 수행 ...
        >>> duration = time.time() - start_time
        >>> log_performance_metrics("Evidence Collector", duration, tokens_used=1500)
    """
    metrics = {
        "duration_seconds": round(duration, 2),
    }
    
    if tokens_used is not None:
        metrics["tokens_used"] = tokens_used
    
    if memory_used is not None:
        metrics["memory_mb"] = round(memory_used, 2)
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    log_agent_activity(
        agent_name=agent_name,
        action="performance_metrics",
        data=metrics,
        level="INFO"
    )


def log_to_json_file(
    agent_name: str,
    action: str,
    data: Optional[Dict[str, Any]],
    level: str
):
    """
    JSON 형식으로 로그 파일에 기록
    
    Args:
        agent_name: 에이전트 이름
        action: 액션
        data: 데이터
        level: 로그 레벨
    """
    json_log_file = LOG_DIR / "agent_activities.jsonl"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "level": level,
        "data": data or {}
    }
    
    try:
        with open(json_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        # 로그 기록 실패는 조용히 무시 (무한 루프 방지)
        print(f"Failed to write JSON log: {e}")


def log_error(
    agent_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
):
    """
    에러 로그 기록
    
    Args:
        agent_name: 에이전트 이름
        error: 예외 객체
        context: 추가 컨텍스트 정보
        
    Examples:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_error("Risk Assessor", e, {"input": "..."})
        ...     raise
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if context:
        error_data["context"] = context
    
    log_agent_activity(
        agent_name=agent_name,
        action="error_occurred",
        data=error_data,
        level="ERROR"
    )


def log_workflow_state(
    workflow_id: str,
    state: str,
    data: Optional[Dict[str, Any]] = None
):
    """
    워크플로우 상태 로그 기록
    
    Args:
        workflow_id: 워크플로우 ID
        state: 현재 상태
        data: 추가 데이터
        
    Examples:
        >>> log_workflow_state(
        ...     workflow_id="workflow_123",
        ...     state="service_profiler_complete",
        ...     data={"next_node": "evidence_collector"}
        ... )
    """
    log_agent_activity(
        agent_name="Workflow",
        action=f"state_transition_{state}",
        data={"workflow_id": workflow_id, **(data or {})}
    )


class PerformanceTimer:
    """성능 측정을 위한 컨텍스트 매니저"""
    
    def __init__(self, agent_name: str, operation: str):
        self.agent_name = agent_name
        self.operation = operation
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        log_agent_activity(
            agent_name=self.agent_name,
            action=f"{self.operation}_start",
            data={"timestamp": self.start_time}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        log_performance_metrics(
            agent_name=self.agent_name,
            duration=duration,
            additional_metrics={"operation": self.operation}
        )
        
        if exc_type is not None:
            log_error(
                agent_name=self.agent_name,
                error=exc_val,
                context={"operation": self.operation, "duration": duration}
            )
        
        return False  # 예외를 다시 발생시킴


def get_performance_summary(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """
    성능 통계 요약 반환
    
    Args:
        agent_name: 특정 에이전트의 통계 (None이면 전체)
        
    Returns:
        성능 통계 딕셔너리
    """
    json_log_file = LOG_DIR / "agent_activities.jsonl"
    
    if not json_log_file.exists():
        return {}
    
    performance_logs = []
    
    with open(json_log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("action") == "performance_metrics":
                    if agent_name is None or log_entry.get("agent") == agent_name:
                        performance_logs.append(log_entry)
            except json.JSONDecodeError:
                continue
    
    if not performance_logs:
        return {}
    
    # 통계 계산
    durations = [log["data"]["duration_seconds"] for log in performance_logs]
    
    summary = {
        "total_operations": len(performance_logs),
        "average_duration": sum(durations) / len(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
    }
    
    # 토큰 사용량 통계
    tokens = [log["data"].get("tokens_used", 0) for log in performance_logs if "tokens_used" in log["data"]]
    if tokens:
        summary["total_tokens"] = sum(tokens)
        summary["average_tokens"] = sum(tokens) / len(tokens)
    
    return summary


# 사용 예시
if __name__ == "__main__":
    # 기본 로깅
    logger = setup_logger("TestAgent")
    logger.info("This is a test log")
    
    # 활동 로깅
    log_agent_activity(
        agent_name="TestAgent",
        action="test_action",
        data={"key": "value"}
    )
    
    # 성능 측정
    with PerformanceTimer("TestAgent", "test_operation"):
        time.sleep(1)  # 시뮬레이션
    
    # 성능 요약
    summary = get_performance_summary("TestAgent")
    print(summary)

