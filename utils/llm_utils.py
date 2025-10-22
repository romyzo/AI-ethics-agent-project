"""
LLM Utility Functions

LLM 호출, 토큰 계산 등 LLM 관련 공통 함수들
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

from .logging_utils import log_agent_activity

# 환경 변수 로드
load_dotenv()


class LLMClient:
    """LLM 클라이언트 래퍼"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.default_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.default_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
    def call_openai(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        OpenAI API 호출 with 에러 핸들링 및 재시도 로직
        
        Args:
            prompt: 입력 프롬프트
            model: 사용할 모델 (기본값: gpt-4o)
            temperature: 온도 파라미터 (기본값: 0.1)
            max_tokens: 최대 토큰 수
            response_format: 응답 형식 (JSON 모드 등)
            
        Returns:
            LLM 응답 텍스트
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.max_tokens
        
        for attempt in range(self.max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.openai_client.chat.completions.create(**kwargs)
                
                # 토큰 사용량 로깅
                log_agent_activity(
                    agent_name="LLM",
                    action="api_call",
                    data={
                        "model": model,
                        "tokens_used": response.usage.total_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                log_agent_activity(
                    agent_name="LLM",
                    action="api_error",
                    data={"error": str(e), "attempt": attempt + 1}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"LLM API 호출 실패 (최대 재시도 횟수 초과): {str(e)}")
    
    def call_anthropic(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Anthropic Claude API 호출
        
        Args:
            prompt: 입력 프롬프트
            model: 사용할 모델 (기본값: claude-3-5-sonnet)
            temperature: 온도 파라미터
            max_tokens: 최대 토큰 수
            
        Returns:
            LLM 응답 텍스트
        """
        model = model or os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022")
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.max_tokens
        
        for attempt in range(self.max_retries):
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # 토큰 사용량 로깅
                log_agent_activity(
                    agent_name="LLM",
                    action="api_call",
                    data={
                        "model": model,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                )
                
                return response.content[0].text
                
            except Exception as e:
                log_agent_activity(
                    agent_name="LLM",
                    action="api_error",
                    data={"error": str(e), "attempt": attempt + 1}
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"Anthropic API 호출 실패: {str(e)}")


# 전역 클라이언트 인스턴스
_llm_client = None

def get_llm_client() -> LLMClient:
    """LLM 클라이언트 싱글톤 반환"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: str = "openai"
) -> str:
    """
    LLM 호출 (에러 핸들링 및 재시도 포함)
    
    Args:
        prompt: 입력 프롬프트
        model: 사용할 모델명
        temperature: 온도 파라미터 (0.0 ~ 1.0)
        max_tokens: 최대 토큰 수
        provider: LLM 제공자 ("openai" 또는 "anthropic")
        
    Returns:
        LLM 응답 텍스트
        
    Examples:
        >>> response = call_llm("Explain AI ethics in 3 sentences")
        >>> print(response)
    """
    client = get_llm_client()
    
    if provider.lower() == "anthropic":
        return client.call_anthropic(prompt, model, temperature, max_tokens)
    else:
        return client.call_openai(prompt, model, temperature, max_tokens)


def call_llm_with_json_output(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    구조화된 JSON 응답을 받는 LLM 호출
    
    Args:
        prompt: 입력 프롬프트 (JSON 형식 응답 요청 포함)
        model: 사용할 모델명
        temperature: 온도 파라미터
        max_tokens: 최대 토큰 수
        
    Returns:
        파싱된 JSON 딕셔너리
        
    Examples:
        >>> prompt = "Analyze this service and return JSON with keys: type, risk_level, description"
        >>> result = call_llm_with_json_output(prompt)
        >>> print(result["risk_level"])
    """
    client = get_llm_client()
    
    # JSON 모드로 호출
    response = client.call_openai(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        log_agent_activity(
            agent_name="LLM",
            action="json_parse_error",
            data={"error": str(e), "response": response[:200]}
        )
        # 응답에서 JSON 부분만 추출 시도
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        raise Exception(f"JSON 파싱 실패: {str(e)}")


def calculate_token_count(text: str, model: str = "gpt-4o") -> int:
    """
    텍스트의 토큰 수 계산
    
    Args:
        text: 토큰 수를 계산할 텍스트
        model: 모델명 (인코딩 방식 결정)
        
    Returns:
        토큰 수
        
    Examples:
        >>> text = "This is a sample text for token counting."
        >>> tokens = calculate_token_count(text)
        >>> print(f"Tokens: {tokens}")
    """
    try:
        # 모델에 따른 인코딩 선택
        if "gpt-4" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)
    
    except Exception as e:
        # 토큰 계산 실패 시 대략적인 추정 (1 토큰 ≈ 4 글자)
        log_agent_activity(
            agent_name="LLM",
            action="token_count_fallback",
            data={"error": str(e)}
        )
        return len(text) // 4


def batch_call_llm(
    prompts: List[str],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: str = "openai"
) -> List[str]:
    """
    여러 프롬프트를 배치로 처리
    
    Args:
        prompts: 프롬프트 리스트
        model: 사용할 모델명
        temperature: 온도 파라미터
        max_tokens: 최대 토큰 수
        provider: LLM 제공자
        
    Returns:
        응답 리스트
    """
    results = []
    for prompt in prompts:
        response = call_llm(prompt, model, temperature, max_tokens, provider)
        results.append(response)
        # Rate limiting 방지
        time.sleep(0.5)
    
    return results


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o"
) -> float:
    """
    API 호출 비용 추정
    
    Args:
        prompt_tokens: 입력 토큰 수
        completion_tokens: 출력 토큰 수
        model: 모델명
        
    Returns:
        예상 비용 (USD)
    """
    # 2024년 기준 대략적인 가격 (실제 가격은 OpenAI 웹사이트 참조)
    pricing = {
        "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        "claude-3-5-sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    }
    
    # 모델에 맞는 가격 찾기
    model_pricing = None
    for key in pricing:
        if key in model.lower():
            model_pricing = pricing[key]
            break
    
    if not model_pricing:
        model_pricing = pricing["gpt-4o"]  # 기본값
    
    cost = (prompt_tokens * model_pricing["input"] + 
            completion_tokens * model_pricing["output"])
    
    return round(cost, 6)

