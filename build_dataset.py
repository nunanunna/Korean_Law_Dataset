"""
국회 법률발의안 API를 활용하여 법안을 가져오고,
제안이유 및 주요내용을 기반으로 10개 카테고리로 분류하여 테스트 데이터셋을 생성하는 스크립트.

카테고리: 노동, 복지, 주거, 경제, 교육, 환경·기후, 디지털, 보건, 생활안전, 정치·행정
각 카테고리당 10개씩 법률을 저장합니다. (총 100개 법안, 다중 카테고리 법안은 초과 가능)
"""

import requests
import json
import re
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ASSEMBLY_API_KEY")
BILL_API_URL = "https://open.assembly.go.kr/portal/openapi/nzmimeepazxkubdpn"
PLENARY_API_URL = "https://open.assembly.go.kr/portal/openapi/nkalemivaqmoibxro"
SUMMARY_URL = "http://likms.assembly.go.kr/bill/bi/popup/billSummary.do"

# 10개 카테고리와 키워드 매핑
CATEGORIES = {
    "노동": {
        "keywords": [
            "근로", "노동", "임금", "고용", "해고", "퇴직", "산업재해", "산재", "직업",
            "취업", "일자리", "파견", "비정규", "정규직", "노조", "노동조합", "단체교섭",
            "최저임금", "근로기준", "작업환경", "직장", "사업주", "사용자", "근무", "야간근로",
            "초과근로", "연장근로", "유급휴가", "연차", "파업", "쟁의", "산업안전", "직업훈련",
            "고용보험", "실업", "워라밸", "주52시간", "플랫폼노동", "특수고용", "프리랜서"
        ],
        "strong_keywords": ["근로", "노동", "임금", "고용보험", "산업재해", "노동조합", "최저임금", "근로기준"]
    },
    "복지": {
        "keywords": [
            "복지", "장애인", "아동", "노인", "보육", "사회보장", "기초생활", "수급",
            "사회서비스", "돌봄", "양육", "출산", "저출산", "육아", "어린이집", "유치원",
            "연금", "국민연금", "기초연금", "장애", "발달장애", "자립지원", "사회적약자",
            "한부모", "다문화", "이민", "외국인", "난민", "취약계층", "빈곤", "저소득",
            "사회복지", "긴급복지", "의료급여", "생활보장", "가정폭력", "아동학대"
        ],
        "strong_keywords": ["복지", "장애인", "사회보장", "기초생활", "돌봄", "사회서비스", "아동학대"]
    },
    "주거": {
        "keywords": [
            "주거", "주택", "임대", "전세", "월세", "부동산", "아파트", "건축",
            "재건축", "재개발", "공공임대", "분양", "청약", "매매", "등기", "토지",
            "도시정비", "리모델링", "건설", "건물", "공동주택", "주택도시기금",
            "공공주택", "택지", "주거급여", "주거안정", "임차인", "임대인", "보증금",
            "전월세", "집값", "부동산거래", "주택공급", "공인중개", "중개사"
        ],
        "strong_keywords": ["주거", "주택", "임대", "전세", "부동산", "재건축", "공공임대"]
    },
    "경제": {
        "keywords": [
            "경제", "금융", "세금", "조세", "소득세", "법인세", "부가가치세", "관세",
            "무역", "수출", "수입", "산업", "기업", "중소기업", "벤처", "창업",
            "공정거래", "독점", "소비자", "물가", "인플레이션", "은행", "보험", "증권",
            "주식", "자본시장", "투자", "외국인투자", "특허", "지식재산", "저작권",
            "통상", "FTA", "상거래", "전자상거래", "규제", "세제", "재정", "예산",
            "국채", "공공기관", "공기업", "민영화", "경쟁", "시장", "소상공인", "자영업",
            "특구", "경제자유구역", "대기업", "재벌", "지주회사"
        ],
        "strong_keywords": ["경제", "금융", "조세", "소득세", "법인세", "공정거래", "중소기업", "소비자"]
    },
    "교육": {
        "keywords": [
            "교육", "학교", "학생", "교사", "교원", "대학", "입시", "수능",
            "학력", "교과서", "교과과정", "방과후", "학원", "사교육", "공교육",
            "장학금", "등록금", "학자금", "특수교육", "영재", "직업교육", "평생교육",
            "교육감", "교육청", "초등학교", "중학교", "고등학교", "유아교육",
            "학교폭력", "교육과정", "원격수업", "이러닝", "학점", "학위"
        ],
        "strong_keywords": ["교육", "학교", "학생", "교사", "교원", "대학", "교육과정"]
    },
    "환경·기후": {
        "keywords": [
            "환경", "기후", "탄소", "온실가스", "배출", "대기오염", "수질", "토양오염",
            "폐기물", "쓰레기", "재활용", "녹색", "신재생에너지", "태양광", "풍력",
            "에너지", "원자력", "발전소", "전기", "가스", "석유", "석탄",
            "자연보호", "생태계", "생물다양성", "국립공원", "산림", "숲",
            "물관리", "수자원", "홍수", "가뭄", "미세먼지", "환경영향평가",
            "탄소중립", "기후변화", "지구온난화", "ESG", "친환경", "녹색성장"
        ],
        "strong_keywords": ["환경", "기후", "탄소", "온실가스", "대기오염", "폐기물", "신재생에너지", "탄소중립"]
    },
    "디지털": {
        "keywords": [
            "디지털", "정보통신", "인터넷", "소프트웨어", "데이터", "인공지능", "AI",
            "블록체인", "핀테크", "가상자산", "암호화폐", "전자서명", "전자정부",
            "정보보호", "개인정보", "사이버", "해킹", "통신", "방송", "미디어",
            "OTT", "플랫폼", "클라우드", "빅데이터", "IoT", "사물인터넷",
            "메타버스", "자율주행", "로봇", "드론", "반도체", "전자", "ICT",
            "스마트", "앱", "모바일", "5G", "6G", "과학기술", "연구개발"
        ],
        "strong_keywords": ["디지털", "인공지능", "AI", "개인정보", "데이터", "정보통신", "가상자산"]
    },
    "보건": {
        "keywords": [
            "보건", "의료", "병원", "의사", "간호", "약사", "약국", "의약품",
            "건강", "질병", "감염", "전염병", "방역", "백신", "치료", "진료",
            "건강보험", "의료보험", "의료기기", "임상시험", "제약", "바이오",
            "정신건강", "자살예방", "중독", "마약", "치매", "암", "희귀질환",
            "응급의료", "구급", "혈액", "장기이식", "한의학", "한방", "의료사고",
            "공중보건", "위생", "식품안전", "식품위생", "영양", "건강검진"
        ],
        "strong_keywords": ["보건", "의료", "병원", "건강보험", "의약품", "감염", "방역"]
    },
    "생활안전": {
        "keywords": [
            "안전", "재난", "소방", "화재", "지진", "태풍", "재해", "방재",
            "교통", "도로", "자동차", "운전", "철도", "항공", "선박", "해양",
            "식품", "위해", "제품안전", "어린이안전", "놀이시설", "승강기",
            "가스안전", "전기안전", "시설물", "건축물안전", "안전관리",
            "치안", "범죄", "경찰", "형사", "처벌", "형법", "성범죄",
            "스토킹", "보이스피싱", "사기", "성폭력", "가정폭력", "학교폭력",
            "CCTV", "112", "119", "민방위", "소비자안전", "체육시설"
        ],
        "strong_keywords": ["안전", "재난", "소방", "교통", "도로", "범죄", "경찰", "재해"]
    },
    "정치·행정": {
        "keywords": [
            "선거", "투표", "국회", "의원", "정당", "대통령", "국무총리", "국무위원",
            "행정", "공무원", "지방자치", "지방의회", "자치단체", "시장", "도지사", "군수",
            "구청장", "주민투표", "주민소환", "조례", "감사원", "감사", "탄핵", "국정조사",
            "국정감사", "청문회", "정부조직", "행정기관", "중앙행정", "지방행정",
            "공직", "공직자윤리", "이해충돌", "정치자금", "선거운동", "후보자",
            "비례대표", "지역구", "선거구", "공직선거", "정보공개", "행정절차",
            "민원", "옴부즈만", "국민권익", "부패방지", "행정소송", "행정심판",
            "인사", "공무원연금", "병역", "국방", "군사", "외교", "통일", "안보",
            "헌법", "법원", "검찰", "사법", "재판", "판사", "변호사"
        ],
        "strong_keywords": ["선거", "국회", "공무원", "행정", "지방자치", "정당", "국정감사", "공직선거"]
    }
}


def fetch_bills(page_size=100, page_index=1, age="22"):
    """국회 법률발의안 API에서 의안 목록 가져오기"""
    params = {
        "Key": API_KEY,
        "Type": "json",
        "pSize": page_size,
        "pIndex": page_index,
        "AGE": age
    }
    try:
        r = requests.get(BILL_API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        key = "nzmimeepazxkubdpn"
        if key in data:
            rows = data[key][1].get("row", [])
            total = data[key][0]["head"][0]["list_total_count"]
            return rows, total
    except Exception as e:
        print(f"  [ERROR] 법률발의안 API 호출 실패: {e}")
    return [], 0


def fetch_plenary_bills(page_size=100, page_index=1, age="21"):
    """국회 본회의 처리안건 API에서 의안 목록 가져오기"""
    params = {
        "Key": API_KEY,
        "Type": "json",
        "pSize": page_size,
        "pIndex": page_index,
        "AGE": age
    }
    try:
        r = requests.get(PLENARY_API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        key = "nkalemivaqmoibxro"
        if key in data:
            rows = data[key][1].get("row", [])
            total = data[key][0]["head"][0]["list_total_count"]
            return rows, total
    except Exception as e:
        print(f"  [ERROR] 본회의 처리안건 API 호출 실패: {e}")
    return [], 0


def fetch_bill_summary(bill_id):
    """의안 상세 페이지에서 제안이유 및 주요내용 가져오기"""
    try:
        r = requests.get(SUMMARY_URL, params={"billId": bill_id}, timeout=30)
        r.encoding = 'utf-8'
        text = r.text
        
        # <pre class="print_pre"> 태그 내용 추출
        match = re.search(r'<pre[^>]*class="print_pre"[^>]*>(.*?)</pre>', text, re.DOTALL)
        if match:
            content = match.group(1)
            # HTML 엔티티 및 태그 정리
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace('&nbsp;', ' ')
            content = content.replace('&lt;', '<')
            content = content.replace('&gt;', '>')
            content = content.replace('&amp;', '&')
            content = re.sub(r'\r\n', '\n', content)
            content = re.sub(r'\n{3,}', '\n\n', content)
            return content.strip()
        
        return None
    except Exception as e:
        print(f"  [ERROR] 제안이유 가져오기 실패 (bill_id={bill_id}): {e}")
        return None


def classify_bill(bill_name, summary_text):
    """법안 이름과 제안이유를 기반으로 카테고리 분류 (최소 1개, 최대 3개)"""
    if not summary_text:
        return []
    
    combined_text = f"{bill_name} {summary_text}"
    
    scores = {}
    for category, info in CATEGORIES.items():
        score = 0
        matched_keywords = []
        
        # 일반 키워드 매칭 (1점)
        for kw in info["keywords"]:
            count = combined_text.count(kw)
            if count > 0:
                score += count
                matched_keywords.append(kw)
        
        # 강한 키워드 매칭 (추가 3점)
        for kw in info["strong_keywords"]:
            count = combined_text.count(kw)
            if count > 0:
                score += count * 3
        
        if score > 0:
            scores[category] = score
    
    if not scores:
        return []
    
    # 점수 기준 정렬
    sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 최대 3개, 최소 1개 선택
    # 1위 대비 30% 이상인 카테고리만 포함
    top_score = sorted_categories[0][1]
    result = []
    for cat, score in sorted_categories[:3]:
        if score >= top_score * 0.3:
            result.append(cat)
    
    return result if result else [sorted_categories[0][0]]


def collect_bills_from_api(categorized_bills, collected_bill_ids, age, max_pages, target):
    """주어진 대수(age)의 API에서 법안을 수집하여 카테고리별로 분류"""
    page_index = 1
    total_fetched = 0
    
    while page_index <= max_pages:
        # 모든 카테고리가 채워졌는지 확인
        all_full = all(len(v) >= target for v in categorized_bills.values())
        if all_full:
            print(f"\n  [OK] 모든 카테고리가 {target}개씩 채워졌습니다!")
            break
        
        unfilled_status = {cat: f"{len(bills)}/{target}" 
                          for cat, bills in categorized_bills.items() 
                          if len(bills) < target}
        print(f"\n  >> {age}대 페이지 {page_index} 수집 중... (미충족: {unfilled_status})")
        
        rows, total = fetch_bills(page_size=100, page_index=page_index, age=age)
        
        if not rows:
            print(f"  [!] 페이지 {page_index}에서 데이터 없음.")
            break
        
        for row in rows:
            # 모든 카테고리가 채워졌으면 중단
            all_full = all(len(v) >= target for v in categorized_bills.values())
            if all_full:
                break
            
            bill_id = row.get("BILL_ID", "")
            bill_name = row.get("BILL_NAME", "")
            bill_no = row.get("BILL_NO", "")
            
            if bill_id in collected_bill_ids:
                continue
            
            # 법률안만 (예산안, 결의안 등 제외)
            if "일부개정" not in bill_name and "전부개정" not in bill_name and "제정" not in bill_name:
                continue
            
            # 제안이유 가져오기
            summary = fetch_bill_summary(bill_id)
            if not summary or len(summary) < 50:
                continue
            
            # 카테고리 분류
            categories = classify_bill(bill_name, summary)
            if not categories:
                continue
            
            # 아직 부족한 카테고리가 있는 법안만 추가
            needs_adding = any(
                len(categorized_bills[cat]) < target 
                for cat in categories
            )
            
            if not needs_adding:
                continue
            
            bill_data = {
                "bill_id": bill_id,
                "bill_no": bill_no,
                "bill_name": bill_name,
                "proposer": row.get("PROPOSER", ""),
                "propose_dt": row.get("PROPOSE_DT", ""),
                "committee": row.get("COMMITTEE", ""),
                "proc_result": row.get("PROC_RESULT", ""),
                "age": row.get("AGE", ""),
                "detail_link": row.get("DETAIL_LINK", ""),
                "summary": summary,
                "categories": categories
            }
            
            # 분류된 모든 카테고리에 추가 (다중 카테고리 허용)
            for cat in categories:
                if len(categorized_bills[cat]) < target:
                    categorized_bills[cat].append(bill_data)
            
            collected_bill_ids.add(bill_id)
            total_fetched += 1
            print(f"    + [{bill_no}] {bill_name} -> {categories}")
            
            # API 부하 방지
            time.sleep(0.3)
        
        page_index += 1
    
    return total_fetched


def main():
    print("=" * 60)
    print("법률 유사도 측정 테스트 데이터셋 생성")
    print(f"카테고리: {', '.join(CATEGORIES.keys())}")
    print(f"목표: 각 카테고리당 10개, 총 100개 법안")
    print("=" * 60)
    
    # 카테고리별 저장할 법안 수
    TARGET_PER_CATEGORY = 10
    
    # 카테고리별 수집된 법안
    categorized_bills = {cat: [] for cat in CATEGORIES}
    
    # 이미 수집된 법안 ID 추적
    collected_bill_ids = set()
    
    max_pages = 50  # 최대 50페이지(5000건)까지 수집
    
    # ========================================
    # 1단계: 22대 국회 데이터 수집
    # ========================================
    print("\n[1단계] 22대 국회 법률발의안 수집 중...")
    fetched = collect_bills_from_api(
        categorized_bills, collected_bill_ids, 
        age="22", max_pages=max_pages, target=TARGET_PER_CATEGORY
    )
    print(f"\n  22대 수집 완료: {fetched}건")
    
    # ========================================
    # 2단계: 부족한 카테고리 보충 (21대)
    # ========================================
    unfilled = {cat: TARGET_PER_CATEGORY - len(bills) 
                for cat, bills in categorized_bills.items() 
                if len(bills) < TARGET_PER_CATEGORY}
    
    if unfilled:
        print(f"\n[2단계] 부족한 카테고리 보충 (21대 국회)...")
        print(f"  부족 현황: {unfilled}")
        fetched = collect_bills_from_api(
            categorized_bills, collected_bill_ids,
            age="21", max_pages=max_pages, target=TARGET_PER_CATEGORY
        )
        print(f"\n  21대 수집 완료: {fetched}건")
    
    # ========================================
    # 3단계: 데이터셋 저장
    # ========================================
    print("\n[3단계] 데이터셋 저장 중...")
    
    os.makedirs("test_dataset", exist_ok=True)
    
    # 카테고리별 JSON 파일 저장
    for cat, bills in categorized_bills.items():
        # 파일명에 사용할 수 있도록 카테고리명 정리
        safe_cat = cat.replace("·", "_")
        filepath = os.path.join("test_dataset", f"{safe_cat}.json")
        
        dataset = {
            "category": cat,
            "count": len(bills[:TARGET_PER_CATEGORY]),
            "bills": bills[:TARGET_PER_CATEGORY]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"  > {filepath}: {len(bills[:TARGET_PER_CATEGORY])}개 법안 저장")
    
    # 전체 통합 데이터셋 저장
    all_bills = []
    for cat, bills in categorized_bills.items():
        for bill in bills[:TARGET_PER_CATEGORY]:
            all_bills.append(bill)
    
    # 중복 제거 (여러 카테고리에 속한 법안)
    seen = set()
    unique_bills = []
    for bill in all_bills:
        if bill["bill_id"] not in seen:
            seen.add(bill["bill_id"])
            unique_bills.append(bill)
    
    full_dataset = {
        "description": "법률 간 유사도 측정을 위한 테스트 데이터셋",
        "categories": list(CATEGORIES.keys()),
        "total_unique_bills": len(unique_bills),
        "total_with_duplicates": sum(len(bills[:TARGET_PER_CATEGORY]) for bills in categorized_bills.values()),
        "category_counts": {cat: len(bills[:TARGET_PER_CATEGORY]) for cat, bills in categorized_bills.items()},
        "bills": unique_bills
    }
    
    with open(os.path.join("test_dataset", "full_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    
    # ========================================
    # 결과 요약
    # ========================================
    print("\n" + "=" * 60)
    print("데이터셋 생성 결과 요약")
    print("=" * 60)
    for cat in CATEGORIES:
        count = len(categorized_bills[cat][:TARGET_PER_CATEGORY])
        status = "[OK]" if count >= TARGET_PER_CATEGORY else "[!!]"
        print(f"  {status} {cat}: {count}/{TARGET_PER_CATEGORY}개")
    
    total_slots = sum(len(bills[:TARGET_PER_CATEGORY]) for bills in categorized_bills.values())
    print(f"\n  카테고리별 합계 (중복 포함): {total_slots}개")
    print(f"  고유 법안 수: {len(unique_bills)}개")
    print(f"  저장 위치: test_dataset/")
    print("=" * 60)


if __name__ == "__main__":
    main()
