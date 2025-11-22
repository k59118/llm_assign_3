import os
import json
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# --- 설정 영역 ---

INPUT_PATH = "je_ko_sent_test.json"      # 원본 파일 경로
OUTPUT_PATH = "je_ko_test.json"         # 결과 파일 경로
MODEL_NAME = "gpt-4.1-mini"              # 원하는 모델명으로 변경 가능
MAX_ITEMS = None                         # None이면 전체, 숫자면 앞에서부터 N개만 처리

# 병렬 처리 설정
BATCH_SIZE = 100                          # 한 번에 최소 30개 이상 처리
MAX_WORKERS = 100                         # 동시에 보낼 쓰레드 수
SLEEP_SEC = 0.3                          # 배치 간 간격 (rate limit 완화용)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- 프롬프트 템플릿 ---

SYSTEM_PROMPT = """당신은 한국어 데이터셋 구축을 돕는 언어학 조수입니다.
표준어 문장(ko)와 제주 방언 문장(je)이 같은 의미를 가진 한 쌍이 주어집니다.

당신의 작업:
1. 두 문장의 의미가 모두 자연스럽게 답이 될 수 있는 '질문'을 표준어로 한 문장 생성하세요.
2. 질문은 ko/je 둘 다로 대답해도 의미가 통하도록 작성해야 합니다.
3. 질문에 ko 문장 내용을 그대로 복사해서 넣지 말고, 의미를 한 단계 추상화해 주세요.
4. 존댓말/반말은 자유롭게 쓰되, 자연스럽고 과하게 장황하지 않게 만드세요.
5. 질문은 한 문장만 출력하세요.
6. 출력은 질문 문장만 그대로 출력하세요. 다른 설명이나 따옴표를 붙이지 마세요.
"""

USER_PROMPT_TEMPLATE = """다음 두 문장은 같은 의미를 가진 문장 쌍입니다.

[표준어(ko)]
{ko}

[제주 방언(je)]
{je}

위 두 문장이 모두 자연스럽게 대답이 될 수 있는 질문을 표준어로 한 문장 만들어 주세요.
질문만 한 문장으로 출력해 주세요.
"""


def generate_question_for_pair(ko: str, je: str) -> str:
    """
    (ko, je) pair에 대해 LLM을 호출하여 question 한 문장을 생성한다.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(ko=ko, je=je)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=64,
    )

    question = response.choices[0].message.content.strip()
    # 혹시 여러 줄이 오면 첫 줄만 사용
    question = question.split("\n")[0].strip()
    return question


def process_single_item(idx: int, item: Dict[str, Any], total: int) -> Tuple[int, Dict[str, Any]]:
    """
    하나의 item을 처리하여 question을 생성하고, (idx, 처리된 item)을 반환한다.
    예외/리트라이 로직을 이 안에서 처리한다.
    """
    processed = dict(item)

    ko = processed.get("ko", "").strip()
    je = processed.get("je", "").strip()

    if not ko or not je:
        processed["question"] = ""
        print(f"[{idx+1}/{total}] SKIP - empty ko/je")
        return idx, processed

    for attempt in range(3):
        try:
            question = generate_question_for_pair(ko, je)
            processed["question"] = question
            print(f"[{idx+1}/{total}] OK - {question}")
            return idx, processed
        except Exception as e:
            print(f"[{idx+1}/{total}] ERROR (attempt {attempt+1}): {e}")
            time.sleep(2)

    processed["question"] = ""
    print(f"[{idx+1}/{total}] FAILED - question empty fallback")
    return idx, processed


def save_results(results: List[Any], data: List[Dict[str, Any]]) -> None:
    """
    현재까지의 results를 파일로 저장한다.
    아직 처리되지 않은 인덱스는 원본 data를 그대로 사용해 길이를 유지한다.
    """
    output: List[Dict[str, Any]] = []
    for i in range(len(data)):
        item = results[i] if results[i] is not None else data[i]
        output.append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    # 입력 데이터 로드
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if MAX_ITEMS is not None:
        data = data[:MAX_ITEMS]

    total = len(data)

    # 기존 출력 파일에서 이미 처리된 인덱스 복원 (재시작/스킵용)
    results: List[Any] = [None] * total
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) == total:
                restored = 0
                for i, item in enumerate(existing):
                    # question이 비어 있지 않은 경우만 "이미 처리됨"으로 간주
                    if isinstance(item, dict) and item.get("question"):
                        results[i] = item
                        restored += 1
                print(f"기존 결과에서 {restored}/{total}개 인덱스를 복원하여 스킵합니다.")
        except Exception as e:
            print(f"기존 출력 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다: {e}")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_indices = list(range(batch_start, batch_end))

        print(f"\n=== Batch {batch_start+1} ~ {batch_end} / {total} (size={len(batch_indices)}) ===")

        # 이미 question이 채워진 인덱스는 스킵
        pending_indices = [idx for idx in batch_indices if results[idx] is None]
        if not pending_indices:
            print("이 배치의 모든 인덱스가 이미 처리되어 스킵합니다.")
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_single_item, idx, data[idx], total): idx
                    for idx in pending_indices
                }

                for future in as_completed(futures):
                    idx, processed_item = future.result()
                    results[idx] = processed_item

        # 배치가 끝날 때마다 중간 결과 저장
        save_results(results, data)

        if batch_end < total and SLEEP_SEC > 0:
            time.sleep(SLEEP_SEC)

    # 한 번 더 최종 저장(안전용)
    save_results(results, data)
    print(f"\n완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
