"""
사용자가 수정한 카테고리(삭제 및 추가)를 정밀 분석하여 동기화하고,
법안 개수가 10개 미만인 카테고리를 채우는 스크립트
"""

import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ASSEMBLY_API_KEY")
BILL_API_URL = "https://open.assembly.go.kr/portal/openapi/nzmimeepazxkubdpn"
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

dataset_dir = "test_dataset"
TARGET_PER_CATEGORY = 10

# 1. 파일 목록 및 수정 시간 로드
files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
mtimes = {filename: os.path.getmtime(os.path.join(dataset_dir, filename)) for filename in files}

# 2. full_dataset.json에서 이전 카테고리 정보 로드 (old_categories로 매핑)
old_categories_map = {}
full_dataset_path = os.path.join(dataset_dir, "full_dataset.json")
if os.path.exists(full_dataset_path):
    with open(full_dataset_path, "r", encoding="utf-8") as f:
        try:
            full_data = json.load(f)
            for b in full_data.get("bills", []):
                old_categories_map[b["bill_id"]] = b.get("categories", [])
        except Exception as e:
            print(f"Warning loading full_dataset.json: {e}")

# 3. 개별 카테고리 파일에서 현재 존재하는 법안 인스턴스 수집
# bill_id -> { filename: bill_dict }
bill_instances = {}
for filename in files:
    if filename == "full_dataset.json":
        continue
    filepath = os.path.join(dataset_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            bills = data.get("bills", [])
            for b in bills:
                bid = b["bill_id"]
                if bid not in bill_instances:
                    bill_instances[bid] = {}
                bill_instances[bid][filename] = b
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# 4. 각 법안에 대한 최종 카테고리 결정 알고리즘 적용
# 규칙:
# - actual_categories: 실제로 발견된 파일들의 카테고리명 목록
# - declared_categories: 발견된 파일들 중 가장 mtime이 늦은 파일에 적혀 있는 categories 필드
# - old_categories: full_dataset.json에 기록되어 있던 이전 카테고리 필드
# - final_categories:
#     1) declared_categories를 기반으로 하되,
#     2) 만약 C가 old_categories에 있었는데 actual_categories에 없다면 (사용자가 C.json에서 지운 것), 최종 카테고리에서 C를 지웁니다.
#     3) 만약 D가 declared_categories에 새로 추가되었고 actual_categories에는 아직 없다면, 사용자가 새로 지정한 것이므로 최종 카테고리에 유지합니다.

final_bills = {}
for bid, instances in bill_instances.items():
    # mtime 기준 정렬하여 가장 최신 파일 찾기
    sorted_instances = sorted(instances.items(), key=lambda x: mtimes[x[0]], reverse=True)
    best_file, best_bill_data = sorted_instances[0]
    
    # 1) declared_categories & actual_categories
    declared_categories = best_bill_data.get("categories", [])
    
    actual_categories = []
    for fname in instances.keys():
        # 파일명에서 카테고리명 복원 (예: 환경_기후.json -> 환경·기후)
        cat_name = fname.replace(".json", "").replace("_", "·")
        actual_categories.append(cat_name)
        
    old_categories = old_categories_map.get(bid, [])
    
    # 2) & 3) 최종 카테고리 결정
    final_cats = set(declared_categories)
    
    # 예전엔 있었는데 실제로 파일 목록에서 지워진 것은 사용자가 삭제한 것임
    for c in old_categories:
        if c not in actual_categories:
            final_cats.discard(c)
            
    # 최종 리스트로 변환
    final_cats_list = sorted(list(final_cats))
    
    bill_info = dict(best_bill_data)
    bill_info["categories"] = final_cats_list
    final_bills[bid] = bill_info
    
    # 변경 사항 감지되면 로그 출력
    if set(old_categories) != set(final_cats_list):
        print(f"Adjusted Categories for '{bill_info['bill_name']}': {old_categories} -> {final_cats_list}")

# 5. 최종 카테고리에 맞춰 법안 재배치
categorized_bills = {cat: [] for cat in CATEGORIES}
for bid, bill_data in final_bills.items():
    cats = bill_data["categories"]
    for cat in cats:
        if cat in categorized_bills:
            categorized_bills[cat].append(bill_data)

# 각 카테고리별 개수 분석
print("\n=== After Reallocation ===")
for cat, bills in categorized_bills.items():
    print(f"  {cat}: {len(bills)} bills")

# 6. 부족한 카테고리 채우기
def fetch_bills_from_api(page_size=100, page_index=1, age="22"):
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
            return rows
    except Exception as e:
        print(f"  [ERROR] API 호출 실패: {e}")
    return []

def fetch_bill_summary(bill_id):
    try:
        r = requests.get(SUMMARY_URL, params={"billId": bill_id}, timeout=30)
        r.encoding = 'utf-8'
        text = r.text
        match = re.search(r'<pre[^>]*class="print_pre"[^>]*>(.*?)</pre>', text, re.DOTALL)
        if match:
            content = match.group(1)
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace('&nbsp;', ' ')
            content = content.replace('&lt;', '<')
            content = content.replace('&gt;', '>')
            content = content.replace('&amp;', '&')
            content = re.sub(r'\r\n', '\n', content)
            content = re.sub(r'\n{3,}', '\n\n', content)
            return content.strip()
    except Exception as e:
        print(f"  [ERROR] 제안이유 가져오기 실패 ({bill_id}): {e}")
    return None

def classify_bill(bill_name, summary_text):
    if not summary_text:
        return []
    combined_text = f"{bill_name} {summary_text}"
    scores = {}
    for category, info in CATEGORIES.items():
        score = 0
        for kw in info["keywords"]:
            count = combined_text.count(kw)
            if count > 0:
                score += count
        for kw in info["strong_keywords"]:
            count = combined_text.count(kw)
            if count > 0:
                score += count * 3
        if score > 0:
            scores[category] = score
    if not scores:
        return []
    sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_categories[0][1]
    result = []
    for cat, score in sorted_categories[:3]:
        if score >= top_score * 0.3:
            result.append(cat)
    return result if result else [sorted_categories[0][0]]

# 이미 존재하는 모든 bill_id 수집
existing_bill_ids = set(final_bills.keys())

print("\n=== Fetching new bills for missing categories ===")
max_pages = 50
ages = ["22", "21"]

for age in ages:
    all_filled = all(len(bills) >= TARGET_PER_CATEGORY for bills in categorized_bills.values())
    if all_filled:
        break
        
    print(f"\nSearching in {age}대 assembly...")
    page_index = 4 if age == "22" else 1 # 이전 수집 내역(3페이지) 이후인 4페이지부터 시작
    
    while page_index <= max_pages:
        all_filled = all(len(bills) >= TARGET_PER_CATEGORY for bills in categorized_bills.values())
        if all_filled:
            break
            
        unfilled_status = {cat: f"{len(bills)}/10" for cat, bills in categorized_bills.items() if len(bills) < 10}
        print(f"  Page {page_index} (unfilled: {unfilled_status})")
        
        rows = fetch_bills_from_api(page_size=100, page_index=page_index, age=age)
        if not rows:
            break
            
        for row in rows:
            all_filled = all(len(bills) >= TARGET_PER_CATEGORY for bills in categorized_bills.values())
            if all_filled:
                break
                
            bill_id = row.get("BILL_ID", "")
            bill_name = row.get("BILL_NAME", "")
            bill_no = row.get("BILL_NO", "")
            
            if bill_id in existing_bill_ids:
                continue
                
            if "일부개정" not in bill_name and "전부개정" not in bill_name and "제정" not in bill_name:
                continue
                
            # 부족한 카테고리가 있는지 확인
            summary = fetch_bill_summary(bill_id)
            if not summary or len(summary) < 50:
                continue
                
            categories = classify_bill(bill_name, summary)
            if not categories:
                continue
                
            # 이 신규 법안이 부족한 카테고리 중 하나라도 채워줄 수 있는지 확인
            useful = any(
                len(categorized_bills[cat]) < TARGET_PER_CATEGORY 
                for cat in categories if cat in categorized_bills
            )
            
            if useful:
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
                
                # 해당하는 모든 카테고리에 추가
                for cat in categories:
                    if cat in categorized_bills and len(categorized_bills[cat]) < TARGET_PER_CATEGORY:
                        categorized_bills[cat].append(bill_data)
                
                existing_bill_ids.add(bill_id)
                final_bills[bill_id] = bill_data
                print(f"    Added [{bill_no}] {bill_name} -> {categories}")
                
            time.sleep(0.3)
            
        page_index += 1

# 7. 데이터셋 다시 저장
print("\n=== Saving Fixed Dataset ===")
for cat, bills in categorized_bills.items():
    safe_cat = cat.replace("·", "_")
    filepath = os.path.join(dataset_dir, f"{safe_cat}.json")
    
    # 10개만 저장하도록 슬라이싱
    final_cat_bills = bills[:TARGET_PER_CATEGORY]
    
    dataset = {
        "category": cat,
        "count": len(final_cat_bills),
        "bills": final_cat_bills
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  Saved {filepath} with {len(final_cat_bills)} bills")

# 전체 고유 법안 수집
all_final_unique_bills = {}
for cat, bills in categorized_bills.items():
    for b in bills[:TARGET_PER_CATEGORY]:
        all_final_unique_bills[b["bill_id"]] = b

full_dataset = {
    "description": "법률 간 유사도 측정을 위한 테스트 데이터셋 (수정본)",
    "categories": list(CATEGORIES.keys()),
    "total_unique_bills": len(all_final_unique_bills),
    "total_with_duplicates": sum(len(bills[:TARGET_PER_CATEGORY]) for bills in categorized_bills.values()),
    "category_counts": {cat: len(bills[:TARGET_PER_CATEGORY]) for cat, bills in categorized_bills.items()},
    "bills": list(all_final_unique_bills.values())
}

with open(os.path.join(dataset_dir, "full_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(full_dataset, f, ensure_ascii=False, indent=2)
print("Saved full_dataset.json")

print("\n=== Verification Summary ===")
for cat, bills in categorized_bills.items():
    print(f"  {cat}: {len(bills[:TARGET_PER_CATEGORY])}/10")
print(f"Total Unique Bills: {len(all_final_unique_bills)}")
