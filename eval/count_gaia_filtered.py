from datasets import load_dataset
from collections import defaultdict

# 제외할 도구 키워드
EXCLUDED_TOOL_KEYWORDS = [
    "Image recognition",
    "PowerPoint viewer",
    "Audio processing",
    "Video processing",
    "OCR",
    "Computer vision",
    "Color recognition",
    "Python IDE",
    'Video'
]

def has_excluded_tools(metadata):
    """Annotator Metadata의 Tools 필드에 제외할 도구가 포함되어 있는지 확인"""
    if not metadata:
        return False
    
    tools_text = metadata.get("Tools", "")
    if not tools_text:
        return False
    
    # Tools 텍스트에 제외할 키워드가 있는지 확인
    tools_lower = tools_text.lower()
    for keyword in EXCLUDED_TOOL_KEYWORDS:
        if keyword.lower() in tools_lower:
            return True
    
    return False

# GAIA dev/validation split 전체 로드
ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

# 1. file_name이 비어 있고
# 2. 제외할 도구를 필요로 하지 않는 샘플만 필터링
def is_web_browsing_only(ex):
    # file_name 체크
    if ex.get("file_name") not in (None, "", "NA"):
        return False
    
    # Tools 체크
    metadata = ex.get("Annotator Metadata", {})
    if has_excluded_tools(metadata):
        return False
    
    return True

text_only = ds.filter(is_web_browsing_only)

# 통계 출력
print("=" * 80)
print("GAIA 데이터셋 필터링 결과")
print("=" * 80)
print(f"전체 샘플: {len(ds)}")
print(f"필터링된 샘플 (파일 없고 제외 도구 없음): {len(text_only)} ({len(text_only)/len(ds)*100:.1f}%)")
print()

# Level별 통계
level_stats = defaultdict(lambda: {"total": 0, "filtered": 0})
for ex in ds:
    level = ex["Level"]
    level_stats[level]["total"] += 1

for ex in text_only:
    level = ex["Level"]
    level_stats[level]["filtered"] += 1

print("Level별 통계:")
for level in ["1", "2", "3"]:
    stats = level_stats[level]
    print(f"  Level {level}: {stats['filtered']}/{stats['total']} ({stats['filtered']/stats['total']*100:.1f}%)")

print("=" * 80)
print()

# task_id 리스트 뽑기
task_ids = [ex["task_id"] for ex in text_only]

# 제외된 샘플 분석
print("제외된 샘플 분석:")
excluded_with_file = 0
excluded_with_tools = 0
excluded_both = 0

for ex in ds:
    if ex["task_id"] in task_ids:
        continue
    
    has_file = ex.get("file_name") not in (None, "", "NA")
    has_tools = has_excluded_tools(ex.get("Annotator Metadata", {}))
    
    if has_file and has_tools:
        excluded_both += 1
    elif has_file:
        excluded_with_file += 1
    elif has_tools:
        excluded_with_tools += 1

print(f"  파일만 있어서 제외: {excluded_with_file}")
print(f"  제외 도구만 있어서 제외: {excluded_with_tools}")
print(f"  둘 다 있어서 제외: {excluded_both}")
print(f"  총 제외: {len(ds) - len(text_only)}")
print()

# 제외 도구가 있는 샘플 예시
print("제외 도구가 포함된 샘플 예시:")
count = 0
for ex in ds:
    if has_excluded_tools(ex.get("Annotator Metadata", {})):
        print(f"\n  Task ID: {ex['task_id']}")
        print(f"  Level: {ex['Level']}")
        print(f"  Tools: {ex.get('Annotator Metadata', {}).get('Tools', 'N/A')[:150]}...")
        count += 1
        if count >= 3:
            break
