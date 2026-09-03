"""evaluation/test_queries.csv 를 읽어 실행하고 evaluation/eval_report.md 를 만드는 평가 스크립트."""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.api import app  # noqa: E402

load_dotenv()

_CSV_PATH = _PROJECT_ROOT / "evaluation" / "test_queries.csv"
_REPORT_PATH = _PROJECT_ROOT / "evaluation" / "eval_report4.md"
_CONTINUATION_MARKERS = ("같은 session_id", "이어지는 후속 턴", "후속 턴")


def _split(value: str, sep: str) -> list[str]:
    """콤마/세미콜론으로 구분된 문자열을 비어있지 않은 항목 리스트로 쪼갠다."""
    return [item.strip() for item in value.split(sep) if item.strip()]


def _load_test_cases(csv_path: Path) -> list[dict]:
    """test_queries.csv를 읽어 각 필드를 파싱한 행 목록으로 반환한다."""
    with csv_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["expected_tools_list"] = _split(row["expected_tools"], ",")
        row["forbidden_list"] = _split(row["forbidden"], ",")
        row["expected_traits_list"] = _split(row["expected_traits"], ";")
    return rows


def _resolve_session_id(row: dict, previous_session_id: str | None) -> str:
    """note에 후속 턴 표시가 있으면 직전 행의 session_id를 재사용하고, 없으면 새로 만든다."""
    is_continuation = any(marker in row["note"] for marker in _CONTINUATION_MARKERS)
    if is_continuation and previous_session_id is not None:
        return previous_session_id
    return str(uuid4())


def _check_tools(actual_trace: list[str], expected_tools: list[str]) -> bool:
    """도구 호출 순서가 기대한 것과 정확히 일치하는지 확인한다."""
    return actual_trace == expected_tools


def _check_forbidden(answer: str, forbidden_terms: list[str]) -> list[str]:
    """답변에 등장하면 안 되는 문자열이 실제로 나왔는지 확인한다. 위반 목록을 반환한다."""
    return [term for term in forbidden_terms if term in answer]


class _TraitVerdict(BaseModel):
    """expected_trait 문장 하나에 대한 LLM 판정."""

    trait: str
    passed: bool
    reason: str = Field(description="판정 근거를 한 문장으로")


class _TraitVerdicts(BaseModel):
    verdicts: list[_TraitVerdict]


def _judge_traits(question: str, answer: str, expected_traits: list[str]) -> list[dict]:
    """expected_traits는 자유 서술이라 같은 Bedrock 모델을 LLM-judge로 재사용해 참고용으로 판정한다.

    이 판정은 자동 채점(PASS/FAIL 집계)에 반영하지 않는다 — 사람이 답변 원문과
    나란히 두고 최종 확인하라는 취지로 리포트에만 기록한다.
    """
    if not expected_traits:
        return []

    from langchain_aws import ChatBedrockConverse

    llm = ChatBedrockConverse(
        model_id=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    ).with_structured_output(_TraitVerdicts)

    prompt = (
        "다음은 여행 플래너 에이전트의 질문과 답변이다. 아래 기대 조건(expected_traits) "
        "각각에 대해 답변이 그 조건을 만족하는지 PASS/FAIL과 근거를 판정하라.\n\n"
        f"질문: {question}\n답변: {answer}\n\n기대 조건:\n"
        + "\n".join(f"- {trait}" for trait in expected_traits)
    )
    result: _TraitVerdicts = llm.invoke(prompt)
    return [verdict.model_dump() for verdict in result.verdicts]


def _run_case(client: TestClient, row: dict, session_id: str) -> dict:
    """한 케이스를 /query로 실행하고 채점 결과를 반환한다."""
    response = client.post("/query", json={"question": row["question"], "session_id": session_id})
    response.raise_for_status()
    body = response.json()

    tools_ok = _check_tools(body["trace"], row["expected_tools_list"])
    forbidden_hits = _check_forbidden(body["answer"], row["forbidden_list"])
    forbidden_ok = not forbidden_hits
    traits = _judge_traits(row["question"], body["answer"], row["expected_traits_list"])

    return {
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "session_id": session_id,
        "answer": body["answer"],
        "trace": body["trace"],
        "expected_tools": row["expected_tools_list"],
        "tools_ok": tools_ok,
        "forbidden_hits": forbidden_hits,
        "forbidden_ok": forbidden_ok,
        "traits": traits,
        "note": row["note"],
        "pass": tools_ok and forbidden_ok,
    }


def _write_report(results: list[dict], out_path: Path) -> None:
    """SERVICE.md 5번 성공 기준 형식으로 요약 표와 상세 결과를 파일에 쓴다."""
    categories = sorted({result["category"] for result in results}, key=lambda c: c)
    lines = [
        "# 해외여행 플랜 Agent 평가 리포트",
        "",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        f"- 사용 모델: {os.environ.get('BEDROCK_MODEL_ID', '(미설정)')}",
        f"- 테스트 케이스: {_CSV_PATH.relative_to(_PROJECT_ROOT)}",
        "",
        "## 요약 (자동 채점: expected_tools 순서 일치 + forbidden 문자열 미포함)",
        "",
        "| category | pass | total | rate |",
        "|---|---|---|---|",
    ]
    total_pass = total_all = 0
    ordered_tool_call_pass = 0
    for category in categories:
        rows = [r for r in results if r["category"] == category]
        passed = sum(1 for r in rows if r["pass"])
        lines.append(f"| {category} | {passed} | {len(rows)} | {round(passed / len(rows) * 100)}% |")
        total_pass += passed
        total_all += len(rows)
    lines.append(f"| **전체** | **{total_pass}** | **{total_all}** | **{round(total_pass / total_all * 100)}%** |")
    lines.append("")

    for result in results:
        if result["expected_tools"] and result["tools_ok"]:
            ordered_tool_call_pass += 1
    three_tool_cases = [r for r in results if r["expected_tools"] == [
        "recommend_destinations", "plan_itinerary", "get_precautions"
    ]]
    three_tool_rate = (
        round(sum(1 for r in three_tool_cases if r["tools_ok"]) / len(three_tool_cases) * 100)
        if three_tool_cases
        else 0
    )
    out_of_scope_fabrications = sum(
        1
        for r in results
        if r["category"] == "negative" and "지어" in "".join(t["reason"] for t in r["traits"] if not t["passed"])
    )

    lines += [
        "## SERVICE.md 5번 성공 기준 대응",
        "",
        f"- 12건 중 10건 이상 통과: {total_pass}/{total_all}건 통과",
        f"- 3개 도구(recommend_destinations→plan_itinerary→get_precautions)가 순서대로 호출된 비율: {three_tool_rate}%",
        f"- negative 케이스에서 지어낸 답변으로 의심되는 건수(참고용, 사람 확인 필요): {out_of_scope_fabrications}건",
        "",
        "> expected_traits는 자동 채점 대상이 아닙니다. 아래 상세 결과에서 답변과 나란히 두었으니 사람이 직접 확인하세요.",
        "",
        "## 상세 결과",
        "",
    ]

    for result in results:
        status = "PASS" if result["pass"] else "FAIL"
        lines.append(f"### {result['id']} [{result['category']}] — {status}")
        lines.append(f"- 질문: {result['question']}")
        lines.append(
            f"- 기대 도구: {','.join(result['expected_tools']) or '(없음)'} / "
            f"실제 호출: {','.join(result['trace']) or '(없음)'} / "
            f"{'일치' if result['tools_ok'] else '불일치'}"
        )
        forbidden_note = "위반 없음" if result["forbidden_ok"] else f"위반: {result['forbidden_hits']}"
        lines.append(f"- forbidden 검사: {forbidden_note}")
        lines.append(f"- 답변: {result['answer']}")
        if result["traits"]:
            trait_lines = "; ".join(
                f"{'✅' if t['passed'] else '❌'} {t['trait']} ({t['reason']})" for t in result["traits"]
            )
            lines.append(f"- expected_traits (사람 확인용): {trait_lines}")
        lines.append(f"- note: {result['note']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(use_llm_judge: bool = True) -> None:
    """전체 테스트 케이스를 실행하고 채점해 리포트를 만든다."""
    global _judge_traits
    if not use_llm_judge:
        original = _judge_traits

        def _skip(question: str, answer: str, expected_traits: list[str]) -> list[dict]:
            return []

        _judge_traits = _skip  # type: ignore[assignment]

    test_cases = _load_test_cases(_CSV_PATH)
    client = TestClient(app)

    results: list[dict] = []
    previous_session_id: str | None = None
    for row in test_cases:
        session_id = _resolve_session_id(row, previous_session_id)
        print(f"[{row['id']}] 실행 중...", flush=True)
        try:
            result = _run_case(client, row, session_id)
        except Exception as error:  # noqa: BLE001 - 케이스 하나의 실패가 전체 리포트 생성을 막지 않게 한다
            result = {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "session_id": session_id,
                "answer": f"(실행 오류: {error})",
                "trace": [],
                "expected_tools": row["expected_tools_list"],
                "tools_ok": False,
                "forbidden_hits": [],
                "forbidden_ok": True,
                "traits": [],
                "note": row["note"],
                "pass": False,
            }
            print(f"[{row['id']}] 오류: {error}", flush=True)
        results.append(result)
        previous_session_id = session_id

    _write_report(results, _REPORT_PATH)
    print(f"평가 완료: {_REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_queries.csv 평가 실행")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="expected_traits에 대한 LLM-judge 호출을 건너뛴다 (tools/forbidden 채점만 수행)",
    )
    args = parser.parse_args()
    run(use_llm_judge=not args.no_llm_judge)
