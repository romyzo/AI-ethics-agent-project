"""
Prompt Template Utility Functions

프롬프트 템플릿 로딩, 포맷팅 관련 함수들
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json

from .logging_utils import log_agent_activity, log_error

# 프롬프트 디렉토리
PROMPTS_DIR = Path("prompts")


def load_prompt_template(template_name: str, format_type: str = "yaml") -> str:
    """
    프롬프트 템플릿 파일 로드
    
    Args:
        template_name: 템플릿 파일명 (확장자 제외 가능)
        format_type: 파일 형식 ("yaml", "txt", "json")
        
    Returns:
        프롬프트 템플릿 문자열
        
    Raises:
        FileNotFoundError: 템플릿 파일이 없을 경우
        
    Examples:
        >>> template = load_prompt_template("service_profiler")
        >>> print(template[:100])
    """
    # 확장자 처리
    if not template_name.endswith(('.yaml', '.yml', '.txt', '.json')):
        if format_type == "yaml":
            template_name += ".yaml"
        elif format_type == "txt":
            template_name += ".txt"
        elif format_type == "json":
            template_name += ".json"
    
    template_path = PROMPTS_DIR / template_name
    
    if not template_path.exists():
        # 대체 경로 시도 (ai_ethics_agent/prompts/)
        alt_path = Path("ai_ethics_agent") / PROMPTS_DIR / template_name
        if alt_path.exists():
            template_path = alt_path
        else:
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            if format_type == "yaml" or template_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
                # YAML에서 'prompt' 키를 찾거나 전체 내용 반환
                if isinstance(data, dict):
                    template = data.get('prompt', data.get('template', str(data)))
                else:
                    template = str(data)
            elif format_type == "json" or template_path.suffix == '.json':
                data = json.load(f)
                template = data.get('prompt', data.get('template', str(data)))
            else:
                template = f.read()
        
        log_agent_activity(
            agent_name="PromptLoader",
            action="template_loaded",
            data={"template": template_name, "length": len(template)}
        )
        
        return template
    
    except Exception as e:
        log_error("PromptLoader", e, {"template": template_name})
        raise


def format_prompt(template: str, **kwargs) -> str:
    """
    프롬프트 템플릿에 변수 채우기
    
    Args:
        template: 프롬프트 템플릿 ({variable} 형식)
        **kwargs: 채울 변수들
        
    Returns:
        포맷팅된 프롬프트
        
    Examples:
        >>> template = "Analyze {service_type} for {risk_category} risks."
        >>> prompt = format_prompt(template, service_type="chatbot", risk_category="privacy")
        >>> print(prompt)
        "Analyze chatbot for privacy risks."
    """
    try:
        formatted = template.format(**kwargs)
        
        log_agent_activity(
            agent_name="PromptFormatter",
            action="prompt_formatted",
            data={
                "template_length": len(template),
                "formatted_length": len(formatted),
                "variables": list(kwargs.keys())
            }
        )
        
        return formatted
    
    except KeyError as e:
        log_error("PromptFormatter", e, {"template": template[:100], "kwargs": list(kwargs.keys())})
        raise ValueError(f"Missing template variable: {e}")


def load_and_format_prompt(template_name: str, **kwargs) -> str:
    """
    프롬프트 템플릿 로드 및 포맷팅 (one-step)
    
    Args:
        template_name: 템플릿 파일명
        **kwargs: 채울 변수들
        
    Returns:
        포맷팅된 프롬프트
        
    Examples:
        >>> prompt = load_and_format_prompt(
        ...     "risk_assessment",
        ...     service_name="AI Chatbot",
        ...     category="bias"
        ... )
    """
    template = load_prompt_template(template_name)
    return format_prompt(template, **kwargs)


def create_system_prompt(role: str, guidelines: Optional[str] = None) -> str:
    """
    시스템 프롬프트 생성
    
    Args:
        role: AI의 역할
        guidelines: 추가 가이드라인
        
    Returns:
        시스템 프롬프트
        
    Examples:
        >>> system_prompt = create_system_prompt(
        ...     role="AI Ethics Risk Assessor",
        ...     guidelines="Focus on bias and fairness issues."
        ... )
    """
    base_prompt = f"You are an expert {role}."
    
    if guidelines:
        base_prompt += f"\n\n{guidelines}"
    
    base_prompt += "\n\nProvide detailed, accurate, and well-structured analysis."
    
    return base_prompt


def create_few_shot_prompt(
    instruction: str,
    examples: list[Dict[str, str]],
    current_input: str
) -> str:
    """
    Few-shot 프롬프트 생성
    
    Args:
        instruction: 작업 설명
        examples: 예시 리스트 [{"input": "...", "output": "..."}, ...]
        current_input: 현재 입력
        
    Returns:
        Few-shot 프롬프트
        
    Examples:
        >>> examples = [
        ...     {"input": "Face recognition app", "output": "High privacy risk"},
        ...     {"input": "Weather app", "output": "Low privacy risk"}
        ... ]
        >>> prompt = create_few_shot_prompt(
        ...     instruction="Assess privacy risk",
        ...     examples=examples,
        ...     current_input="Health tracking app"
        ... )
    """
    prompt_parts = [instruction, ""]
    
    for idx, example in enumerate(examples, 1):
        prompt_parts.append(f"Example {idx}:")
        prompt_parts.append(f"Input: {example['input']}")
        prompt_parts.append(f"Output: {example['output']}")
        prompt_parts.append("")
    
    prompt_parts.append("Now, analyze the following:")
    prompt_parts.append(f"Input: {current_input}")
    prompt_parts.append("Output:")
    
    return "\n".join(prompt_parts)


def validate_prompt(prompt: str, max_length: int = 100000) -> bool:
    """
    프롬프트 유효성 검사
    
    Args:
        prompt: 검사할 프롬프트
        max_length: 최대 길이
        
    Returns:
        유효하면 True
        
    Examples:
        >>> is_valid = validate_prompt("Short prompt")
        >>> print(is_valid)  # True
    """
    if not prompt or not prompt.strip():
        log_agent_activity(
            agent_name="PromptValidator",
            action="validation_failed",
            data={"reason": "Empty prompt"}
        )
        return False
    
    if len(prompt) > max_length:
        log_agent_activity(
            agent_name="PromptValidator",
            action="validation_failed",
            data={"reason": "Prompt too long", "length": len(prompt)}
        )
        return False
    
    return True


def list_available_templates() -> list[str]:
    """
    사용 가능한 프롬프트 템플릿 목록 반환
    
    Returns:
        템플릿 파일명 리스트
        
    Examples:
        >>> templates = list_available_templates()
        >>> print(templates)
    """
    if not PROMPTS_DIR.exists():
        return []
    
    templates = []
    for file_path in PROMPTS_DIR.glob("*"):
        if file_path.suffix in ['.yaml', '.yml', '.txt', '.json']:
            templates.append(file_path.name)
    
    return sorted(templates)


def save_prompt_template(template_name: str, content: str, format_type: str = "yaml"):
    """
    프롬프트 템플릿 저장
    
    Args:
        template_name: 템플릿 파일명
        content: 프롬프트 내용
        format_type: 파일 형식
        
    Examples:
        >>> save_prompt_template(
        ...     "new_template",
        ...     "Analyze {input} for risks.",
        ...     format_type="txt"
        ... )
    """
    PROMPTS_DIR.mkdir(exist_ok=True)
    
    if not template_name.endswith(('.yaml', '.yml', '.txt', '.json')):
        if format_type == "yaml":
            template_name += ".yaml"
        elif format_type == "txt":
            template_name += ".txt"
        elif format_type == "json":
            template_name += ".json"
    
    template_path = PROMPTS_DIR / template_name
    
    try:
        with open(template_path, 'w', encoding='utf-8') as f:
            if format_type == "yaml":
                yaml.dump({"prompt": content}, f, allow_unicode=True)
            elif format_type == "json":
                json.dump({"prompt": content}, f, ensure_ascii=False, indent=2)
            else:
                f.write(content)
        
        log_agent_activity(
            agent_name="PromptSaver",
            action="template_saved",
            data={"template": template_name, "path": str(template_path)}
        )
    
    except Exception as e:
        log_error("PromptSaver", e, {"template": template_name})
        raise


def create_chain_of_thought_prompt(task: str, steps: list[str]) -> str:
    """
    Chain-of-Thought 프롬프트 생성
    
    Args:
        task: 수행할 작업
        steps: 사고 단계들
        
    Returns:
        CoT 프롬프트
        
    Examples:
        >>> steps = [
        ...     "1. Identify the AI service type",
        ...     "2. List potential ethical risks",
        ...     "3. Assess severity of each risk"
        ... ]
        >>> prompt = create_chain_of_thought_prompt("Assess AI ethics risks", steps)
    """
    prompt = f"Task: {task}\n\n"
    prompt += "Let's approach this step by step:\n\n"
    
    for step in steps:
        prompt += f"{step}\n"
    
    prompt += "\nNow, let's begin the analysis:"
    
    return prompt


def extract_variables_from_template(template: str) -> list[str]:
    """
    템플릿에서 변수명 추출
    
    Args:
        template: 프롬프트 템플릿
        
    Returns:
        변수명 리스트
        
    Examples:
        >>> template = "Analyze {service} for {risk_type} risks"
        >>> vars = extract_variables_from_template(template)
        >>> print(vars)  # ['service', 'risk_type']
    """
    import re
    
    # {variable} 형식의 변수 찾기
    pattern = r'\{(\w+)\}'
    variables = re.findall(pattern, template)
    
    return list(set(variables))  # 중복 제거


# 테스트 코드
if __name__ == "__main__":
    print("Testing Prompt Utils...")
    
    # 디렉토리 생성
    PROMPTS_DIR.mkdir(exist_ok=True)
    
    # 샘플 템플릿 생성
    sample_template = """
You are an AI Ethics Risk Assessor.

Analyze the following AI service:
Service Type: {service_type}
Description: {description}

Assess the following risk categories:
{risk_categories}

Provide a detailed risk assessment.
"""
    
    save_prompt_template("sample_assessment", sample_template, "txt")
    print("✓ Sample template saved")
    
    # 템플릿 로드 및 포맷팅
    loaded = load_prompt_template("sample_assessment", "txt")
    formatted = format_prompt(
        loaded,
        service_type="Chatbot",
        description="Customer service AI",
        risk_categories="bias, privacy, transparency"
    )
    print("✓ Template loaded and formatted")
    print(formatted[:200])
    
    # 변수 추출
    variables = extract_variables_from_template(sample_template)
    print(f"✓ Variables found: {variables}")

