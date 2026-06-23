#!/usr/bin/env python3
"""
article_similarity_utils.py
===========================
법안의 개정 조문(안 제O조)들 간의 자카드 유사도를 계산합니다.
"""

def jaccard_similarity(a: list[str], b: list[str]) -> float:
    """
    두 리스트 간의 자카드 유사도를 계산합니다.
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    set_a = set(a)
    set_b = set(b)
    
    # 공백이나 비어있는 문자열 제거
    set_a = {x.strip() for x in set_a if x and x.strip()}
    set_b = {x.strip() for x in set_b if x and x.strip()}
    
    if not set_a or not set_b:
        return 0.0
        
    union = set_a.union(set_b)
    intersection = set_a.intersection(set_b)
    
    if not union:
        return 0.0
        
    return len(intersection) / len(union)


def compute_article_similarity(article_numbers_a: list[str], article_numbers_b: list[str]) -> float:
    """
    두 개정 조문 목록 간의 유사도를 계산합니다.
    1. 둘 중 하나라도 비어 있으면 0.0을 반환합니다.
    2. 조문 간 자카드 유사도(Jaccard Similarity)를 계산합니다.
    """
    if not article_numbers_a or not article_numbers_b:
        return 0.0
        
    # 자카드 유사도 계산 (완전 일치 조문 기준)
    sim = jaccard_similarity(article_numbers_a, article_numbers_b)
    
    # TODO: 추후 인접 조문 유사도 보정 로직을 추가할 예정입니다.
    # 예: 제5조와 제6조 등 인접성 점수를 부여하여 부분 점수 반영
    
    return sim


if __name__ == "__main__":
    # 간단한 테스트
    a = ["제5조", "제6조", "제7조의2"]
    b = ["제6조", "제7조의2", "제9조"]
    print("Test Jaccard Similarity:", jaccard_similarity(a, b))
    print("Test Article Similarity:", compute_article_similarity(a, b))
