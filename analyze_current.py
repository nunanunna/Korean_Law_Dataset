import os
import json

CATEGORIES = ["노동", "복지", "주거", "경제", "교육", "환경·기후", "디지털", "보건", "생활안전", "정치·행정"]
dataset_dir = "test_dataset"

files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]

# 각 파일의 mtime 확인
mtimes = {f: os.path.getmtime(os.path.join(dataset_dir, f)) for f in files}
print("=== File mtimes ===")
for fname in sorted(files, key=lambda x: mtimes[x], reverse=True):
    print(f"  {fname}: {mtimes[fname]}")

# 각 파일에 존재하는 법안들의 bill_id 목록 및 내부 categories 정보 수집
# bill_id -> { 'bill_name': ..., 'present_in_files': {filename: categories_inside} }
bills_map = {}

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
                if bid not in bills_map:
                    bills_map[bid] = {
                        "bill_name": b["bill_name"],
                        "bill_no": b["bill_no"],
                        "present_in_files": {}
                    }
                bills_map[bid]["present_in_files"][filename] = b.get("categories", [])
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# 상세 리포트 작성
report = []
report.append("=== CURRENT STATE OF DATASET ===")
for bid, info in bills_map.items():
    report.append(f"\nBill: {info['bill_name']} ({info['bill_no']}) [ID: {bid}]")
    for fname, cats in info["present_in_files"].items():
        report.append(f"  - Present in {fname} with categories field: {cats}")

with open("current_dataset_state.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nReport written to current_dataset_state.txt")
