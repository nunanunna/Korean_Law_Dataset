#!/usr/bin/env python3
"""
clean_text_utils.py
===================
국회 법률발의안 요약 텍스트 정제 및 키워드 추출 전처리 모듈
"""

import re
import sys

# Windows 한글 출력 설정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 제거 대상 형식 표현 (Boilerplate phrases)
BOILERPLATE_PATTERNS = [
    r"제안이유\s+및\s+주요내용",
    r"제안이유",
    r"주요내용",
    r"현행법은",
    r"현행법에서는",
    r"현행법상",
    r"그런데",
    r"그러나",
    r"하지만",
    r"다만",
    r"이에",
    r"따라서",
    r"또한",
    r"한편",
    r"현재",
    r"규정하고\s+있음",
    r"규정하고\s+있으나",
    r"필요성이\s+있음",
    r"필요성이\s+제기됨",
    r"우려됨",
    r"지적되고\s+있음",
    r"개정하려는\s+것임",
    r"개정하고자\s+함",
    r"마련하려는\s+것임",
    r"하고자\s+함",
    r"하려는\s+것임"
]

# 절대 제거하지 않는 법적 효과 단어
LEGAL_KEYWORDS = [
    "지원", "제한", "금지", "허용", "의무", "책무", "처벌", "벌칙", "과태료",
    "신고", "선정", "인증", "포상", "확대", "축소", "강화", "완화", "신설",
    "삭제", "개정", "보장", "보호", "관리", "감독", "조사", "공개", "제공",
    "부과", "감면", "지급"
]

# TF-IDF 불용어 (일반 표현)
KEYWORD_STOPWORDS = [
    "위하여", "위해", "대한", "대하여", "통하여", "관하여", "관련하여",
    "등을", "등의", "등에", "하는", "하도록", "하고", "하며", "있는", "없는",
    "경우", "사항", "내용", "필요", "법률안", "일부개정법률안"
]

# 가벼운 키워드 추출용 확장 불용어
LIGHT_STOPWORDS = KEYWORD_STOPWORDS + [
    "및", "등", "안", "제", "이", "그", "저", "것", "수", "할", "한", "로", "으로",
    "을", "를", "은", "는", "가", "이", "의", "와", "과", "에", "게", "에게", "에서",
    "부터", "까지", "고", "라고", "며", "면서", "면", "라면", "도", "만", "나", "이나"
]


def remove_boilerplate_phrases(text: str) -> str:
    """
    국회 발의안에 반복적으로 등장하는 형식적 표현을 제거합니다.
    단, 법적 효과 단어는 제거하지 않습니다.
    """
    if not text:
        return ""
    
    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        # 단, 해당 패턴이 법적 효과 단어를 포함하는 경우의 예외 처리는 정규식 수준에서 
        # 통째로 지우는 형식이므로, 안전하게 해당 상투 문체 표현을 삭제합니다.
        cleaned = re.sub(pattern, "", cleaned)
        
    return cleaned


def normalize_legal_text_for_sbert(text: str) -> str:
    """
    SBERT 입력용 정제 함수.
    형식 표현 제거, 과도한 줄바꿈 제거, 중복 공백 제거를 수행하되,
    자연스러운 문장 구조(조사, 어미 등)는 최대한 유지합니다.
    """
    if not text:
        return ""
    
    # 1. 형식 표현 제거
    cleaned = remove_boilerplate_phrases(text)
    
    # 2. 줄바꿈 제거 (공백으로 치환)
    cleaned = re.sub(r'[\r\n]+', ' ', cleaned)
    
    # 3. 중복 공백 제거 및 trim
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def normalize_legal_text_for_keywords(text: str) -> str:
    """
    TF-IDF 입력용 정제 함수.
    형식 표현 제거, 불필요한 조사/접속/종결 표현을 적극적으로 제거합니다.
    단, 법률 의미를 바꿀 수 있는 단어(법적 효과 단어)는 제거하지 않습니다.
    """
    if not text:
        return ""
    
    # 1. 형식 표현 제거
    cleaned = remove_boilerplate_phrases(text)
    
    # 2. 줄바꿈 제거
    cleaned = re.sub(r'[\r\n]+', ' ', cleaned)
    
    # 3. 어절 단위 필터링
    words = cleaned.split()
    filtered_words = []
    
    for w in words:
        should_remove = False
        # 불용어가 어절 내에 포함되어 있는지 검사
        for stopword in KEYWORD_STOPWORDS:
            if stopword in w:
                # 단, 법적 효과를 나타내는 핵심 단어가 함께 포함되어 있다면 살림
                # 예: "의무를" -> "을"이 포함되지만 "의무"가 있으므로 보존
                #     "개정하도록" -> "하도록"이 불용어지만 "개정"이 있으므로 보존
                has_legal_keyword = any(lk in w for lk in LEGAL_KEYWORDS)
                if not has_legal_keyword:
                    should_remove = True
                    break
        
        if not should_remove:
            filtered_words.append(w)
            
    return " ".join(filtered_words)


def extract_light_keywords(text: str) -> str:
    """
    형태소 분석기 없이 규칙 기반으로 명사/키워드성 문자열을 추출합니다.
    1. 기호 제거 (한글, 숫자, 영문 제외 공백 치환)
    2. 1글자 토큰 제거
    3. 불용어 제거
    4. 중복 토큰은 빈도 반영을 위해 순서 유지
    """
    if not text:
        return ""
        
    # 1. 기호를 공백으로 치환
    cleaned = re.sub(r'[^가-힣A-Za-z0-9]', ' ', text)
    
    # 2. 토큰 분리
    tokens = cleaned.split()
    
    # 3. 필터링
    result_tokens = []
    for t in tokens:
        # 1글자 토큰 제거
        if len(t) <= 1:
            continue
            
        # 불용어 매칭 검사
        if t in LIGHT_STOPWORDS:
            # 법적 효과 단어가 단어에 포함되어 있으면 보존
            if not any(lk in t for lk in LEGAL_KEYWORDS):
                continue
                
        result_tokens.append(t)
        
    return " ".join(result_tokens)


if __name__ == "__main__":
    test_text = """
    제안이유 및 주요내용
    현행법은 장애인에 대한 복지 지원을 규정하고 있으나, 지원의 실효성이 제기됨에 따라 우려가 지적되고 있음.
    이에 장애인의 교육 보장 및 책무를 강화하여 복지 혜택을 제공하고, 이를 위반할 시 벌칙을 부과하고 과태료를 감면하고자 함.
    """
    print("=" * 60)
    print("원래 텍스트:")
    print(test_text.strip())
    print("-" * 60)
    print("remove_boilerplate_phrases:")
    print(remove_boilerplate_phrases(test_text).strip())
    print("-" * 60)
    print("normalize_legal_text_for_sbert:")
    print(normalize_legal_text_for_sbert(test_text))
    print("-" * 60)
    print("normalize_legal_text_for_keywords:")
    print(normalize_legal_text_for_keywords(test_text))
    print("-" * 60)
    print("extract_light_keywords:")
    print(extract_light_keywords(test_text))
    print("=" * 60)
