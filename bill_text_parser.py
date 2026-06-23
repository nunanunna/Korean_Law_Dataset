"""
국회 발의안 summary 텍스트 파서 모듈
"""

import re

def normalize_summary(text: str) -> str:
    """
    텍스트를 정규화합니다.
    - 맨 앞의 "제안이유 및 주요내용" 제거
    - 줄바꿈, 탭, 연속된 공백을 단일 공백으로 치환하고 앞뒤 공백 제거
    """
    if not text:
        return ""
    # 맨 앞의 "제안이유 및 주요내용" 문구 제거 (공백 허용)
    cleaned = re.sub(r'^\s*제안이유\s+및\s+주요내용\s*', '', text)
    # 줄바꿈, 공백, 탭 정규화
    normalized = re.sub(r'\s+', ' ', cleaned).strip()
    return normalized

def extract_article_text(text: str) -> tuple[str, str]:
    """
    텍스트에서 (안 제... ) 형태의 괄호 문장을 추출하여 분리합니다.
    반환값: (괄호가 제거된 본문 텍스트, 추출된 조문 괄호 텍스트)
    """
    if not text:
        return "", ""
    
    # (안 제... ) 형태의 괄호 문장 매칭
    pattern = r'\(\s*안\s+제[^)]+\)'
    match = re.search(pattern, text)
    
    if match:
        article_text = match.group(0)
        # 본문에서 해당 괄호 문장을 제거
        remaining_text = text.replace(article_text, '')
        # 제거 후 다중 공백 정규화 및 마침표 앞 공백 제거
        remaining_text = re.sub(r'\s+', ' ', remaining_text).strip()
        remaining_text = re.sub(r'\s+\.', '.', remaining_text)
        return remaining_text, article_text
    else:
        return text, ""

def extract_article_numbers(text: str) -> list[str]:
    """
    article_text에서 조문 번호를 정규식으로 추출하여 리스트로 반환합니다.
    예: "(안 제5조 등)" -> ["제5조"]
    """
    if not text:
        return []
    
    # "제[숫자]조" 또는 "제[숫자]조의[숫자]" 패턴 추출
    pattern = r'제\s*\d+\s*조(?:\s*의\s*\d+)?'
    matches = re.findall(pattern, text)
    
    # 조문 번호 내부에 불필요한 공백 제거 (예: "제 5 조" -> "제5조")
    cleaned_matches = [re.sub(r'\s+', '', m) for m in matches]
    return cleaned_matches

def split_summary_sections(summary: str) -> dict:
    """
    요약문을 규칙 기반으로 current_law, problem, proposal, article_text, article_numbers로 분리합니다.
    """
    if not summary:
        return {
            "full_text": "",
            "current_law": "",
            "problem": "",
            "proposal": "",
            "article_text": "",
            "article_numbers": []
        }
        
    normalized = normalize_summary(summary)
    remaining_text, article_text = extract_article_text(normalized)
    article_numbers = extract_article_numbers(article_text)
    
    # 1. proposal 분리 ("이에" 기준)
    # 문장 시작점(텍스트 시작 또는 문장 종결 기호 + 공백) 뒤에 나오는 "이에 " 패턴 검색
    ie_match = re.search(r'(?:^|[\.\?\!]\s+)이에\s+', remaining_text)
    if ie_match:
        # 매칭된 "이에"의 실제 시작 인덱스를 특정
        match_str = ie_match.group(0)
        ie_relative_pos = match_str.find("이에")
        ie_index = ie_match.start() + ie_relative_pos
        
        proposal = remaining_text[ie_index:].strip()
        before_ie = remaining_text[:ie_index].strip()
    else:
        proposal = ""
        before_ie = remaining_text
        
    # 2. current_law와 problem 분리 (before_ie 내에서 "그런데", "그러나", "하지만", "다만" 기준)
    problem_keywords = ["그런데", "그러나", "하지만", "다만"]
    first_keyword_pos = -1
    
    for kw in problem_keywords:
        pos = before_ie.find(kw)
        if pos != -1:
            if first_keyword_pos == -1 or pos < first_keyword_pos:
                first_keyword_pos = pos
                
    if first_keyword_pos != -1:
        current_law = before_ie[:first_keyword_pos].strip()
        problem = before_ie[first_keyword_pos:].strip()
    else:
        current_law = before_ie.strip()
        problem = ""
        
    return {
        "full_text": summary,
        "current_law": current_law,
        "problem": problem,
        "proposal": proposal,
        "article_text": article_text,
        "article_numbers": article_numbers
    }
