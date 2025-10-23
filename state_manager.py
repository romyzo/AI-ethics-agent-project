# State 관리 모듈
import json
import os
from datetime import datetime
from typing import Dict, Any

# State 파일 경로 설정
STATE_FILE = "agent_state.json"
OUTPUT_DIR = "output"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# 디렉토리 생성
os.makedirs(REPORTS_DIR, exist_ok=True)

def initialize_state():
    """State 초기화"""
    initial_state = {
        "service_description": "",
        "service_profile": {},
        "evidence_data": {},
        "assessment_result": {},
        "recommendation_result": [],
        "final_report_markdown": "",
        "status": {
            "service_profiler": "pending",
            "evidence_collector": "pending", 
            "risk_assessor": "pending",
            "mitigation_recommender": "pending",
            "report_composer": "pending"
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(initial_state, f, ensure_ascii=False, indent=2)
    
    print(f"✅ State 초기화 완료 → {STATE_FILE}")
    return initial_state

def load_state():
    """State 로드"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"✅ State 로드 완료 → {STATE_FILE}")
        return state
    else:
        print("⚠️ State 파일이 없습니다. 초기화를 진행합니다.")
        return initialize_state()

def save_state(state):
    """State 저장"""
    state["updated_at"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"✅ State 저장 완료 → {STATE_FILE}")

def save_report_to_file(report_content, service_name):
    """최종 보고서를 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"AI_Ethics_Report_{service_name}_{timestamp}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✅ 보고서 저장 완료 → {filepath}")
    return filepath

def update_status(state, agent_name, status):
    """에이전트 상태 업데이트"""
    state["status"][agent_name] = status
    save_state(state)
    print(f"✅ {agent_name} 상태 업데이트: {status}")

# State 초기화
current_state = initialize_state()
