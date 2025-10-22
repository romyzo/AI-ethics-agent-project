"""
Validation Utility Functions

데이터 검증, 품질 체크, 신뢰도 계산 관련 함수들
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .logging_utils import log_agent_activity


def validate_risk_assessment(assessment: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    리스크 평가 결과 검증
    
    Args:
        assessment: 리스크 평가 딕셔너리
        
    Returns:
        (유효성 여부, 에러 메시지 리스트)
        
    Examples:
        >>> assessment = {
        ...     "risk_level": "high",
        ...     "risk_score": 0.8,
        ...     "categories": ["bias", "privacy"]
        ... }
        >>> is_valid, errors = validate_risk_assessment(assessment)
        >>> print(is_valid)
    """
    errors = []
    
    # 필수 필드 확인
    required_fields = ["risk_level", "risk_score", "categories"]
    for field in required_fields:
        if field not in assessment:
            errors.append(f"Missing required field: {field}")
    
    # risk_level 값 검증
    valid_levels = ["low", "medium", "high", "critical"]
    if "risk_level" in assessment:
        if assessment["risk_level"].lower() not in valid_levels:
            errors.append(f"Invalid risk_level: {assessment['risk_level']}. Must be one of {valid_levels}")
    
    # risk_score 범위 검증
    if "risk_score" in assessment:
        score = assessment["risk_score"]
        if not isinstance(score, (int, float)) or not (0 <= score <= 1):
            errors.append(f"Invalid risk_score: {score}. Must be between 0 and 1")
    
    # categories 검증
    if "categories" in assessment:
        if not isinstance(assessment["categories"], list):
            errors.append("categories must be a list")
        elif len(assessment["categories"]) == 0:
            errors.append("categories cannot be empty")
    
    is_valid = len(errors) == 0
    
    log_agent_activity(
        agent_name="Validator",
        action="risk_assessment_validated",
        data={"is_valid": is_valid, "num_errors": len(errors)}
    )
    
    return is_valid, errors


def calculate_confidence_score(data: Dict[str, Any]) -> float:
    """
    결과의 신뢰도 점수 계산
    
    다음 요소를 고려:
    - 데이터 완전성
    - 일관성
    - 증거 품질
    
    Args:
        data: 평가 데이터
        
    Returns:
        신뢰도 점수 (0.0 ~ 1.0)
        
    Examples:
        >>> data = {
        ...     "risk_assessment": {...},
        ...     "evidence_count": 10,
        ...     "source_quality": 0.9
        ... }
        >>> confidence = calculate_confidence_score(data)
        >>> print(f"Confidence: {confidence:.2f}")
    """
    score_components = []
    
    # 1. 데이터 완전성 점수 (40%)
    completeness_score = calculate_completeness_score(data)
    score_components.append(("completeness", completeness_score, 0.4))
    
    # 2. 일관성 점수 (30%)
    consistency_score = calculate_consistency_score(data)
    score_components.append(("consistency", consistency_score, 0.3))
    
    # 3. 증거 품질 점수 (30%)
    evidence_score = calculate_evidence_quality_score(data)
    score_components.append(("evidence", evidence_score, 0.3))
    
    # 가중 평균 계산
    total_score = sum(score * weight for _, score, weight in score_components)
    
    log_agent_activity(
        agent_name="Validator",
        action="confidence_calculated",
        data={
            "total_score": round(total_score, 3),
            "components": {name: round(score, 3) for name, score, _ in score_components}
        }
    )
    
    return round(total_score, 3)


def calculate_completeness_score(data: Dict[str, Any]) -> float:
    """
    데이터 완전성 점수 계산
    
    Args:
        data: 평가 데이터
        
    Returns:
        완전성 점수 (0.0 ~ 1.0)
    """
    expected_fields = [
        "service_profile",
        "evidence",
        "risk_assessment",
        "recommendations"
    ]
    
    present_fields = sum(1 for field in expected_fields if field in data and data[field])
    completeness = present_fields / len(expected_fields)
    
    return completeness


def calculate_consistency_score(data: Dict[str, Any]) -> float:
    """
    데이터 일관성 점수 계산
    
    Args:
        data: 평가 데이터
        
    Returns:
        일관성 점수 (0.0 ~ 1.0)
    """
    consistency_checks = []
    
    # risk_level과 risk_score의 일관성
    if "risk_assessment" in data:
        assessment = data["risk_assessment"]
        if "risk_level" in assessment and "risk_score" in assessment:
            level = assessment["risk_level"].lower()
            score = assessment["risk_score"]
            
            # 레벨과 점수가 일치하는지 확인
            level_ranges = {
                "low": (0, 0.3),
                "medium": (0.3, 0.6),
                "high": (0.6, 0.85),
                "critical": (0.85, 1.0)
            }
            
            if level in level_ranges:
                min_score, max_score = level_ranges[level]
                is_consistent = min_score <= score <= max_score
                consistency_checks.append(is_consistent)
    
    # 증거 수와 평가의 상세도 일관성
    if "evidence" in data and "risk_assessment" in data:
        evidence_count = len(data["evidence"]) if isinstance(data["evidence"], list) else 0
        # 증거가 많을수록 상세한 평가를 기대
        has_detailed_assessment = len(str(data["risk_assessment"])) > 100
        
        if evidence_count >= 3:
            consistency_checks.append(has_detailed_assessment)
    
    if not consistency_checks:
        return 0.8  # 기본값
    
    return sum(consistency_checks) / len(consistency_checks)


def calculate_evidence_quality_score(data: Dict[str, Any]) -> float:
    """
    증거 품질 점수 계산
    
    Args:
        data: 평가 데이터
        
    Returns:
        증거 품질 점수 (0.0 ~ 1.0)
    """
    if "evidence" not in data:
        return 0.0
    
    evidence = data["evidence"]
    
    if not isinstance(evidence, list) or len(evidence) == 0:
        return 0.0
    
    quality_factors = []
    
    # 1. 증거 수량 (적정 수: 5-15개)
    count = len(evidence)
    if 5 <= count <= 15:
        quantity_score = 1.0
    elif count < 5:
        quantity_score = count / 5
    else:
        quantity_score = max(0.5, 1.0 - (count - 15) * 0.05)
    quality_factors.append(quantity_score)
    
    # 2. 증거 다양성 (다양한 소스)
    if all(isinstance(e, dict) for e in evidence):
        sources = set(e.get("source", "unknown") for e in evidence)
        diversity_score = min(1.0, len(sources) / 3)  # 3개 이상의 다른 소스
        quality_factors.append(diversity_score)
    
    # 3. 증거 상세도
    if all(isinstance(e, dict) for e in evidence):
        avg_length = sum(len(str(e.get("content", ""))) for e in evidence) / count
        detail_score = min(1.0, avg_length / 200)  # 평균 200자 이상
        quality_factors.append(detail_score)
    
    return sum(quality_factors) / len(quality_factors)


def sanitize_input(text: str, max_length: int = 100000) -> str:
    """
    입력값 정제 (악의적 입력 제거)
    
    Args:
        text: 입력 텍스트
        max_length: 최대 길이
        
    Returns:
        정제된 텍스트
        
    Examples:
        >>> dirty = "<script>alert('xss')</script>Hello"
        >>> clean = sanitize_input(dirty)
        >>> print(clean)  # "Hello"
    """
    if not text:
        return ""
    
    # 길이 제한
    text = text[:max_length]
    
    # HTML/스크립트 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 특수 제어 문자 제거
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    # SQL 인젝션 패턴 제거 (기본적인 것만)
    dangerous_patterns = [
        r';\s*DROP\s+TABLE',
        r';\s*DELETE\s+FROM',
        r'UNION\s+SELECT',
        r'<script',
        r'javascript:',
        r'onerror\s*=',
        r'onclick\s*='
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def validate_service_profile(profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    서비스 프로파일 검증
    
    Args:
        profile: 서비스 프로파일 딕셔너리
        
    Returns:
        (유효성 여부, 에러 메시지 리스트)
    """
    errors = []
    
    # 필수 필드
    required_fields = ["service_name", "service_type", "description"]
    for field in required_fields:
        if field not in profile or not profile[field]:
            errors.append(f"Missing or empty required field: {field}")
    
    # 서비스 타입 검증
    valid_types = [
        "chatbot", "recommendation", "classification", "prediction",
        "nlp", "computer_vision", "speech", "generative", "other"
    ]
    
    if "service_type" in profile:
        service_type = profile["service_type"].lower()
        if service_type not in valid_types:
            # 경고만 하고 에러는 아님
            log_agent_activity(
                agent_name="Validator",
                action="service_type_warning",
                data={"type": service_type, "valid_types": valid_types}
            )
    
    # 설명 길이 검증
    if "description" in profile:
        desc_length = len(profile["description"])
        if desc_length < 10:
            errors.append("Description is too short (minimum 10 characters)")
        elif desc_length > 50000:
            errors.append("Description is too long (maximum 50,000 characters)")
    
    return len(errors) == 0, errors


def validate_recommendations(recommendations: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    완화 권고안 검증
    
    Args:
        recommendations: 권고안 리스트
        
    Returns:
        (유효성 여부, 에러 메시지 리스트)
    """
    errors = []
    
    if not isinstance(recommendations, list):
        errors.append("recommendations must be a list")
        return False, errors
    
    if len(recommendations) == 0:
        errors.append("recommendations cannot be empty")
        return False, errors
    
    for idx, rec in enumerate(recommendations):
        if not isinstance(rec, dict):
            errors.append(f"Recommendation {idx} must be a dictionary")
            continue
        
        # 필수 필드
        required_fields = ["risk_category", "recommendation", "priority"]
        for field in required_fields:
            if field not in rec:
                errors.append(f"Recommendation {idx} missing field: {field}")
        
        # priority 검증
        if "priority" in rec:
            valid_priorities = ["low", "medium", "high", "critical"]
            if rec["priority"].lower() not in valid_priorities:
                errors.append(f"Recommendation {idx} has invalid priority: {rec['priority']}")
    
    return len(errors) == 0, errors


def check_data_freshness(timestamp: str, max_age_hours: int = 24) -> bool:
    """
    데이터 신선도 확인
    
    Args:
        timestamp: ISO 형식 타임스탬프
        max_age_hours: 최대 허용 시간 (시간 단위)
        
    Returns:
        신선하면 True
    """
    try:
        data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        current_time = datetime.now(data_time.tzinfo)
        
        age_hours = (current_time - data_time).total_seconds() / 3600
        
        is_fresh = age_hours <= max_age_hours
        
        log_agent_activity(
            agent_name="Validator",
            action="freshness_check",
            data={"age_hours": round(age_hours, 2), "is_fresh": is_fresh}
        )
        
        return is_fresh
    
    except Exception as e:
        log_agent_activity(
            agent_name="Validator",
            action="freshness_check_error",
            data={"error": str(e)}
        )
        return False


def validate_report(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    최종 보고서 검증
    
    Args:
        report: 보고서 딕셔너리
        
    Returns:
        (유효성 여부, 에러 메시지 리스트)
    """
    errors = []
    
    # 필수 섹션
    required_sections = [
        "executive_summary",
        "service_profile",
        "risk_assessment",
        "recommendations"
    ]
    
    for section in required_sections:
        if section not in report or not report[section]:
            errors.append(f"Missing required section: {section}")
    
    # 하위 검증
    if "service_profile" in report:
        is_valid, profile_errors = validate_service_profile(report["service_profile"])
        errors.extend(profile_errors)
    
    if "risk_assessment" in report:
        is_valid, assessment_errors = validate_risk_assessment(report["risk_assessment"])
        errors.extend(assessment_errors)
    
    if "recommendations" in report:
        is_valid, rec_errors = validate_recommendations(report["recommendations"])
        errors.extend(rec_errors)
    
    return len(errors) == 0, errors


# 테스트 코드
if __name__ == "__main__":
    print("Testing Validation Utils...")
    
    # 리스크 평가 검증 테스트
    valid_assessment = {
        "risk_level": "high",
        "risk_score": 0.75,
        "categories": ["bias", "privacy"]
    }
    
    is_valid, errors = validate_risk_assessment(valid_assessment)
    print(f"✓ Valid assessment: {is_valid}, Errors: {errors}")
    
    # 신뢰도 점수 계산 테스트
    sample_data = {
        "service_profile": {"name": "Test"},
        "evidence": [{"source": "web", "content": "..." * 50}] * 5,
        "risk_assessment": valid_assessment,
        "recommendations": [{"risk_category": "bias", "recommendation": "Test", "priority": "high"}]
    }
    
    confidence = calculate_confidence_score(sample_data)
    print(f"✓ Confidence score: {confidence}")
    
    # 입력 정제 테스트
    dirty_input = "<script>alert('xss')</script>This is a test"
    clean_input = sanitize_input(dirty_input)
    print(f"✓ Sanitized input: {clean_input}")

