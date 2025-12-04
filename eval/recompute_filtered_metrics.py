#!/usr/bin/env python3
"""
필터링된 task_ids로 기존 결과에서 Pass@1 재계산
"""

import json
from collections import defaultdict
from datasets import load_dataset

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

def is_web_browsing_only(ex):
    # file_name 체크
    if ex.get("file_name") not in (None, "", "NA"):
        return False
    
    # Tools 체크
    metadata = ex.get("Annotator Metadata", {})
    if has_excluded_tools(metadata):
        return False
    
    return True

# 1. 필터링된 task_ids 추출
print("=" * 80)
print("Step 1: 필터링된 task_ids 추출")
print("=" * 80)

ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
text_only = ds.filter(is_web_browsing_only)
filtered_task_ids = set([ex["task_id"] for ex in text_only])

print(f"필터링된 샘플 수: {len(filtered_task_ids)}")
print()

# 2. 결과 파일 로드
print("=" * 80)
print("Step 2: 기존 결과 파일 로드")
print("=" * 80)

results_file = "/home/robin/minimal-web-browser/gaia_results_qwen3_4b_instruct_2507/gaia_results_20251119_230054.json"
results_file = "/home/robin/minimal-web-browser/gaia_search_r1_reproduce_results/gaia_results_20251120_220855.json"
with open(results_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"모델: {data['model']}")
print(f"전체 결과 수: {len(data['results'])}")
print()

# 3. 필터링된 task_ids만 추출하여 재계산
print("=" * 80)
print("Step 3: 필터링된 결과로 Pass@1 재계산")
print("=" * 80)

filtered_results = [r for r in data['results'] if r['task_id'] in filtered_task_ids]
print(f"필터링 후 결과 수: {len(filtered_results)}")
print()

# Level별 통계 계산
level_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for result in filtered_results:
    level = result['level']
    level_stats[level]["total"] += 1
    if result['correct']:
        level_stats[level]["correct"] += 1

# 결과 출력
print("=" * 80)
print("📊 필터링된 결과 Pass@1 메트릭")
print("=" * 80)
print()

total_correct = 0
total_samples = 0

for level in ["1", "2", "3"]:
    if level in level_stats:
        stats = level_stats[level]
        correct = stats["correct"]
        total = stats["total"]
        pass_at_1 = (correct / total * 100) if total > 0 else 0
        
        print(f"Level {level}:")
        print(f"  Pass@1: {pass_at_1:.2f}% ({correct}/{total})")
        print()
        
        total_correct += correct
        total_samples += total

# 평균 계산
avg_pass_at_1 = (total_correct / total_samples * 100) if total_samples > 0 else 0
print(f"Average:")
print(f"  Pass@1: {avg_pass_at_1:.2f}% ({total_correct}/{total_samples})")
print()

print("=" * 80)
print("📊 비교: 원본 vs 필터링")
print("=" * 80)
print()

print("원본 결과 (전체 165개 샘플):")
for level in ["1", "2", "3"]:
    original = data['metrics'][f"Level {level}"]
    print(f"  Level {level}: {original['pass@1']:.2f}% ({original['correct']}/{original['total']})")
print(f"  Average: {data['metrics']['Average']['pass@1']:.2f}% ({data['metrics']['Average']['correct']}/{data['metrics']['Average']['total']})")
print()

print(f"필터링 결과 (웹 브라우징만 {total_samples}개 샘플):")
for level in ["1", "2", "3"]:
    if level in level_stats:
        stats = level_stats[level]
        correct = stats["correct"]
        total = stats["total"]
        pass_at_1 = (correct / total * 100) if total > 0 else 0
        print(f"  Level {level}: {pass_at_1:.2f}% ({correct}/{total})")
print(f"  Average: {avg_pass_at_1:.2f}% ({total_correct}/{total_samples})")
print()

# 개선율 계산
print("=" * 80)
print("📈 개선율")
print("=" * 80)
print()

original_avg = data['metrics']['Average']['pass@1']
improvement = avg_pass_at_1 - original_avg
print(f"평균 Pass@1 변화: {original_avg:.2f}% → {avg_pass_at_1:.2f}% ({improvement:+.2f}%p)")
print()

print("=" * 80)

