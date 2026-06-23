"""
임베딩 입력용 구조화 텍스트 생성 모듈
"""

import json

def format_categories(categories) -> str:
    """
    categories가 list, string, dict 등 다양한 형태일 수 있으므로
    안전하게 문자열로 변환하여 반환합니다.
    """
    if not categories:
        return "[내용 없음]"
    
    if isinstance(categories, list):
        cleaned = [str(c).strip() for c in categories if c]
        return ", ".join(cleaned) if cleaned else "[내용 없음]"
        
    elif isinstance(categories, dict):
        items = [f"{k}: {v}" for k, v in categories.items()]
        return ", ".join(items) if items else "[내용 없음]"
        
    elif isinstance(categories, str):
        return categories.strip() if categories.strip() else "[내용 없음]"
        
    else:
        return str(categories).strip() if str(categories).strip() else "[내용 없음]"

def build_raw_text(bill: dict) -> str:
    """
    Baseline 텍스트 생성
    bill_name + summary를 합친 텍스트를 반환합니다.
    """
    if not bill:
        return "[내용 없음]"
        
    bill_name = bill.get('bill_name', '').strip()
    summary = bill.get('summary', '').strip()
    
    parts = []
    if bill_name:
        parts.append(f"법안명: {bill_name}")
    if summary:
        parts.append(f"제안이유 및 주요내용: {summary}")
        
    result = "\n".join(parts).strip()
    return result if result else "[내용 없음]"

def build_structured_text(bill: dict, sections: dict) -> str:
    """
    구조화 라벨을 붙인 텍스트 생성
    """
    if not bill or not sections:
        return "[내용 없음]"
        
    bill_name = bill.get('bill_name', '').strip() or "[내용 없음]"
    
    current_law = sections.get('current_law', '').strip() or "[내용 없음]"
    problem = sections.get('problem', '').strip() or "[내용 없음]"
    proposal = sections.get('proposal', '').strip() or "[내용 없음]"
    
    # 조문 내용 구성
    article_text = sections.get('article_text', '').strip()
    article_numbers_list = sections.get('article_numbers', [])
    article_numbers = ", ".join(article_numbers_list) if article_numbers_list else ""
    
    if article_text and article_numbers:
        article_info = f"{article_text} (조문번호: {article_numbers})"
    elif article_text:
        article_info = article_text
    elif article_numbers:
        article_info = f"(조문번호: {article_numbers})"
    else:
        article_info = "[내용 없음]"
        
    structured = (
        f"[법안명]\n{bill_name}\n\n"
        f"[현행법]\n{current_law}\n\n"
        f"[문제점]\n{problem}\n\n"
        f"[개정내용]\n{proposal}\n\n"
        f"[개정조문]\n{article_info}"
    )
    return structured

def build_problem_proposal_text(bill: dict, sections: dict) -> str:
    """
    문제점과 개정내용(제안사항)만 결합한 텍스트 생성
    """
    if not bill or not sections:
        return "[내용 없음]"
        
    problem = sections.get('problem', '').strip() or "[내용 없음]"
    proposal = sections.get('proposal', '').strip() or "[내용 없음]"
    
    if problem == "[내용 없음]" and proposal == "[내용 없음]":
        return "[내용 없음]"
        
    result = (
        f"[문제점]\n{problem}\n\n"
        f"[개정내용]\n{proposal}"
    )
    return result

def build_article_text(bill: dict, sections: dict) -> str:
    """
    조문 텍스트와 조문 번호만 결합한 텍스트 생성
    """
    if not bill or not sections:
        return "[내용 없음]"
        
    article_text = sections.get('article_text', '').strip() or "[내용 없음]"
    article_numbers_list = sections.get('article_numbers', [])
    article_numbers = ", ".join(article_numbers_list) if article_numbers_list else "[내용 없음]"
    
    if article_text == "[내용 없음]" and article_numbers == "[내용 없음]":
        return "[내용 없음]"
        
    result = (
        f"[조문내용]\n{article_text}\n\n"
        f"[조문번호]\n{article_numbers}"
    )
    return result
