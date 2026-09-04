"""해외여행 플랜 Agent의 메인 LangGraph 그래프.

슬롯(기간·목적·예산·도시) 추출과 최종 답변 생성에는 langchain.agents의
create_agent를 쓰지만, 여행지 추천 → 일정·예산 플래너 → 주의사항 조회의
호출 순서는 여기서 그래프 엣지로 직접 고정한다 — 에이전트의 자유
tool-calling 루프에 순서를 맡기지 않는다.
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.tools import (
    get_attractions,
    get_precautions,
    list_supported_cities,
    plan_itinerary,
    recommend_destinations,
    resolve_city_name,
)

load_dotenv()

_MAX_RECOMMENDATIONS = 3  # 답변에서 예산·주의사항까지 상세히 보여줄 후보 도시 수


class Slots(TypedDict, total=False):
    """대화에서 채워야 하는 여행 조건."""

    season: str | None
    nights: int | None
    purpose: list[str]
    budget_krw: int | None
    city: str | None


class SlotExtraction(BaseModel):
    """사용자 발화에서 뽑아낸 여행 조건 (구조화 출력용)."""

    season: str | None = Field(None, description="봄/여름/가을/겨울 중 하나. 알 수 없으면 null")
    nights: int | None = Field(None, description="여행 박수(숫자). 알 수 없으면 null")
    purpose: list[str] = Field(
        default_factory=list,
        description="휴양/관광/쇼핑 중 해당하는 것만 골라 담는다. 셋 중 어디에도 안 맞으면 빈 리스트로 둔다",
    )
    budget_krw: int | None = Field(None, description="1인 기준 총예산(원화, 숫자만). 알 수 없으면 null")
    city: str | None = Field(None, description="사용자가 이미 지정한 여행 도시명. 없으면 null")


class SpecialIntent(BaseModel):
    """전체 여행 계획이 아니라 부분 정보만으로도 처리 가능한 특수 의도 분류 (구조화 출력용).

    기간·목적·예산이 다 안 채워진 메시지 중, 곧바로 되묻지 않고 바로 처리해야
    하는 케이스(지원 범위 밖 지명, 안전정보 단독 질문, 강제 예산 배분 요구)를
    가려내기 위한 용도다. 지원 도시인지 여부는 여기서 판단하지 않는다.
    """

    intent: Literal["travel_plan", "precaution_only", "forced_budget_override"] = Field(
        description=(
            "travel_plan: 통상적인 여행 계획 요청(정보가 더 필요하면 되물어야 함). "
            "precaution_only: 여행 계획 전체가 아니라 특정 국가/도시의 비자·안전·치안 "
            "정보만 알고 싶어함. forced_budget_override: 총예산 대비 항목별(항공권/숙박/"
            "교통 등) 금액을 사용자가 직접 못박아 그대로 반영하라고 요구함"
        )
    )
    mentioned_place: str | None = Field(
        None,
        description=(
            "사용자가 언급한 지명. 국가명은 빼고 도시명만 담아라 "
            "(예: '베트남 하노이'가 아니라 '하노이'). 도시가 특정되지 않고 "
            "국가만 언급됐으면 국가명을 담아도 된다. 없으면 null"
        ),
    )
    precaution_target_country: str | None = Field(
        None,
        description="intent가 precaution_only일 때, 언급된 도시를 국가명으로 변환한 값. 특정할 수 없으면 null",
    )


class FollowupIntent(BaseModel):
    """이미 여행지 추천을 받은 세션에서, 후속 메시지의 의도를 분류한다 (구조화 출력용).

    사용자가 이전에 제시된 후보 중 하나를 고르며 관광지·맛집 정보를 원하는
    것인지, 아니면 조건을 바꿔 다시 추천·계획을 받고 싶어하는 것인지 구분한다.
    """

    wants_city_details: bool = Field(
        description=(
            "사용자가 이전 후보 중 특정 도시를 고르며 관광지·꼭 가봐야 할 곳·맛집 "
            "정보를 원하면 true. 예산·기간·목적 등 조건을 바꿔서 다시 추천·계획을 "
            "받고 싶어하는 것이면(재계획 요청) false"
        )
    )
    selected_city: str | None = Field(
        None,
        description=(
            "wants_city_details가 true일 때 사용자가 고른 도시명. 이름으로 "
            "골랐든('오사카') 순번으로 골랐든('1번', '첫번째') 이전 후보 목록과 "
            "대조해서 정확한 도시명 하나로 담아라. 특정할 수 없으면 null"
        ),
    )


class AgentState(TypedDict):
    """그래프 전체에서 공유하는 상태."""

    messages: Annotated[list[BaseMessage], add_messages]
    slots: Slots
    missing_slot: str | None
    destinations_result: dict | None
    itinerary_result: dict | None
    itinerary_results: list[dict]
    precautions_result: dict | None
    precautions_by_country: dict[str, dict]
    attractions_result: dict | None
    special_intent: dict | None
    trace: list[str]
    answer: str
    contexts: list[dict]


_SLOT_QUESTIONS = {
    "기간": "언제, 며칠 동안 여행하실 예정인가요? (예: '10월 5일부터 9일까지' 또는 '가을에 5박6일')",
    "목적": "이번 여행의 목적은 무엇인가요? 휴양·관광·쇼핑 중에서 말씀해 주세요.",
    "예산": "1인 기준 총예산은 얼마로 생각하고 계신가요? (원화 기준)",
}


def _llm() -> ChatBedrockConverse:
    """.env 설정으로 Bedrock 대화 모델을 만든다."""
    return ChatBedrockConverse(
        model_id=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _merge_slots(previous: Slots, extracted: SlotExtraction) -> Slots:
    """새로 추출한 값으로 이전 슬롯을 덮어쓰되, 비어 있으면(None) 이전 값을 유지한다."""
    merged: Slots = dict(previous)
    if extracted.season is not None:
        merged["season"] = extracted.season
    if extracted.nights is not None:
        merged["nights"] = extracted.nights
    if extracted.purpose:
        merged["purpose"] = extracted.purpose
    if extracted.budget_krw is not None:
        merged["budget_krw"] = extracted.budget_krw
    if extracted.city is not None:
        # "베트남 하노이"처럼 국가명이 섞여 와도 지원 도시명("하노이")으로 정규화한다.
        # 정규화가 안 되면(진짜 미지원 지명이면) 추출된 값을 그대로 둔다.
        merged["city"] = resolve_city_name(extracted.city) or extracted.city
    merged.setdefault("purpose", [])
    return merged


def _slots_for_prompt(slots: Slots) -> dict:
    """슬롯을 답변 생성 프롬프트에 넘기기 전에, 예산을 미리 '만원' 단위 문자열로
    포맷해 budget_krw_formatted로 덧붙인다 — LLM이 원 단위 정수를 직접
    만원으로 환산하다 자릿수를 틀리는 경우(예: 150만원을 1,500만원으로 표기)가
    있어, 계산을 코드에서 끝내고 넘겨준다.
    """
    slots_dict = dict(slots)
    budget_krw = slots.get("budget_krw")
    if budget_krw:
        slots_dict["budget_krw_formatted"] = f"{budget_krw // 10000}만원"
    return slots_dict


def _missing_slot(slots: Slots) -> str | None:
    """기간 → 목적 → 예산 순으로, 아직 안 채워진 필수 슬롯을 하나만 고른다."""
    if not slots.get("season") or not slots.get("nights"):
        return "기간"
    if not slots.get("purpose"):
        return "목적"
    if not slots.get("budget_krw"):
        return "예산"
    return None


def extract_slots_node(state: AgentState) -> dict:
    """사용자 발화를 구조화 추출해 이전 슬롯과 병합한다.

    매 턴의 첫 노드라 여기서 trace를 [] 로 초기화한다 — trace는 세션
    체크포인트에 그대로 보존되는 값이라, 여기서 리셋하지 않으면 이전 턴에
    호출된 도구까지 이번 턴 trace에 계속 쌓여버린다(턴 내부에서 여러
    노드를 거치며 누적되는 동작 자체는 그대로 유지됨).
    """
    slot_agent = create_agent(
        model=_llm(),
        tools=None,
        response_format=SlotExtraction,
        system_prompt=(
            "사용자의 여행 계획 발화에서 여행 기간(계절/박수), 목적, 예산, 도시를 "
            "최대한 뽑아내라. 목적은 휴양/관광/쇼핑 중에서만 고르고, 셋 중 무엇에도 "
            "해당하지 않으면 빈 리스트로 둬라. 모르는 값은 null로 남기고 임의로 "
            "단정하지 마라."
        ),
    )
    result = slot_agent.invoke({"messages": state["messages"]})
    extracted: SlotExtraction = result["structured_response"]
    slots = _merge_slots(state.get("slots", {}), extracted)
    return {"slots": slots, "missing_slot": _missing_slot(slots), "trace": []}


def route_after_extract(state: AgentState) -> Literal["ask_missing_slot", "recommend"]:
    """필수 슬롯이 부족하면 되묻고, 다 채워졌으면 추천 단계로 간다."""
    return "ask_missing_slot" if state["missing_slot"] else "recommend"


def route_after_extract_extended(
    state: AgentState,
) -> Literal[
    "recommend", "classify_followup", "scope_check", "classify_partial_intent", "ask_missing_slot"
]:
    """슬롯이 다 찼으면 기존과 동일하게 진행하고, 부족할 때만 특수 의도(지원 범위 밖
    지명·안전정보 단독 질문·강제 예산 배분 요구)를 먼저 확인한다.

    기존 route_after_extract를 대체하지 않고 그래프 배선에서만 이 함수를 쓴다 —
    슬롯이 이미 다 채워진 경우(P01~P05, N02, E02)의 동작은 route_after_extract와
    100% 동일하다. 단, 이전 턴에 이미 추천을 준 세션(destinations_result가 있음)
    이면 후속 의도(관광지·맛집 상세 vs 재계획)를 먼저 확인한다.
    """
    if state["missing_slot"] is None:
        if state.get("destinations_result") is not None:
            return "classify_followup"
        return "recommend"

    city = (state.get("slots") or {}).get("city")
    if city and city not in list_supported_cities():
        return "scope_check"

    if state["missing_slot"] == "기간":
        # 기간(날짜) 정보 자체가 없는 메시지만 특수 의도 분류 대상으로 삼는다 —
        # 예산만/목적만 없는 정상 후속 질문(E01, E03)은 추가 LLM 호출 없이
        # 기존과 동일하게 바로 되묻는다.
        return "classify_partial_intent"

    return "ask_missing_slot"


def classify_partial_intent_node(state: AgentState) -> dict:
    """기간 정보가 없는 메시지가 통상적인 여행 계획 요청인지, 아니면 부분 정보만으로
    바로 처리해야 하는 특수 의도(안전정보 단독 질문, 강제 예산 배분 요구 등)인지 분류한다.
    """
    intent_agent = create_agent(
        model=_llm(),
        tools=None,
        response_format=SpecialIntent,
        system_prompt=(
            "사용자 메시지가 통상적인 여행 계획 요청(travel_plan)인지, 특정 국가/도시의 "
            "비자·안전·치안 정보만 묻는 것인지(precaution_only), 아니면 총예산을 넘는 "
            "항목별 금액을 그대로 반영하라고 요구하는 것인지(forced_budget_override) "
            "분류하라. 언급된 지명이 실제 지원 대상인지는 판단하지 말고, mentioned_place에 "
            "담아라 — 단 국가명은 빼고 도시명만 담아라(예: '베트남 하노이'가 아니라 "
            "'하노이'). precaution_only면 그 지명이 속한 국가명을 precaution_target_country에 "
            "담아라(모르면 null)."
        ),
    )
    result = intent_agent.invoke({"messages": state["messages"]})
    extracted: SpecialIntent = result["structured_response"]
    intent_data = extracted.model_dump()
    if intent_data.get("mentioned_place"):
        # 프롬프트를 안 따르고 "베트남 하노이"처럼 통짜로 와도 도시명만 뽑아 정규화한다.
        intent_data["mentioned_place"] = (
            resolve_city_name(intent_data["mentioned_place"]) or intent_data["mentioned_place"]
        )
    return {"special_intent": intent_data}


def route_after_classify(
    state: AgentState,
) -> Literal["scope_check", "precaution_only", "budget_override_guard", "ask_missing_slot"]:
    """분류된 특수 의도에 따라 전용 노드로 보내고, 통상적인 요청이면 기존 되묻기로 합류시킨다."""
    intent = state.get("special_intent") or {}
    place = intent.get("mentioned_place")
    if place and place not in list_supported_cities():
        return "scope_check"
    if intent.get("intent") == "precaution_only":
        return "precaution_only"
    if intent.get("intent") == "forced_budget_override":
        return "budget_override_guard"
    return "ask_missing_slot"


def scope_check_node(state: AgentState) -> dict:
    """언급된 지명이 지원 범위(4개국 12개 도시) 밖인지 확인한다.

    기간·목적이 없어도 안전하게 동작하는 recommend_destinations를 그대로
    재사용한다 — season/purpose가 비어 있어도 필터링만 실패할 뿐이며,
    도시가 미지원 목록 밖이면 그것만으로 곧바로 in_scope=False가 된다.
    """
    slots = state.get("slots") or {}
    intent = state.get("special_intent") or {}
    place = slots.get("city") or intent.get("mentioned_place")
    result = recommend_destinations(
        season=slots.get("season") or "", purpose=slots.get("purpose") or [], city=place
    )
    return {
        "destinations_result": result,
        "trace": state.get("trace", []) + ["recommend_destinations"],
    }


def precaution_only_node(state: AgentState) -> dict:
    """여행 계획 전체가 아니라 안전정보만 묻는 질문에 곧바로 주의사항 조회 도구를 호출한다."""
    intent = state.get("special_intent") or {}
    country = intent.get("precaution_target_country")
    result = get_precautions(country=country)
    return {
        "precautions_result": result,
        "trace": state.get("trace", []) + ["get_precautions"],
        "contexts": result["contexts"],
    }


def budget_override_guard_node(state: AgentState) -> dict:
    """사용자가 강제한 항목별 예산 배분을 실제로 반영하지 않는다는 것을 보여준다.

    city/season/nights가 모두 있으면 실제 값으로 일정·예산 플래너를 호출해
    정상적인(사용자가 부른 금액이 아닌) 배분을 제시하고, 하나라도 없으면
    plan_itinerary가 안전하게 "가격 데이터 없음"으로 끝나는 placeholder 값을
    넣는다 — plan_itinerary 자체가 항목별 강제 배분을 받는 인자를 두고 있지
    않다는 사실(구조적 가드레일)을 그대로 활용한다.
    """
    slots = state.get("slots") or {}
    season, nights, city = slots.get("season"), slots.get("nights"), slots.get("city")
    if season and nights and city:
        base = plan_itinerary(
            city=city, season=season, nights=nights, total_budget_krw=slots.get("budget_krw") or 0
        )
    else:
        # 셋 중 하나라도 없으면 우연히 매치되는 일이 없도록 전부 placeholder로 둔다
        base = plan_itinerary(city="", season="", nights=0, total_budget_krw=slots.get("budget_krw") or 0)

    enriched = dict(base)
    enriched["forced_allocation_rejected"] = True
    enriched["policy_note"] = (
        "사용자가 직접 지정한 항목별 금액의 합계가 총예산을 넘더라도, 그 지정 금액을 "
        "그대로 반영하지 않고 총예산 이내로만 배분한다는 정책에 따라 요청을 그대로 "
        "따르지 않았다."
    )
    return {
        "itinerary_result": enriched,
        "trace": state.get("trace", []) + ["plan_itinerary"],
    }


def classify_followup_node(state: AgentState) -> dict:
    """이미 추천을 받은 세션의 후속 메시지가 도시 상세 정보(관광지·맛집) 요청인지,
    조건을 바꾼 재계획 요청인지 분류한다. 직전 후보 목록을 프롬프트에 넣어줘서
    "1번"/"첫번째" 같은 순번 표현도 정확한 도시명으로 바로 뽑아내게 한다.
    """
    candidates = state["destinations_result"].get("candidates", [])
    candidate_list = ", ".join(
        f"{i + 1}. {candidate['city']}" for i, candidate in enumerate(candidates)
    )
    intent_agent = create_agent(
        model=_llm(),
        tools=None,
        response_format=FollowupIntent,
        system_prompt=(
            "직전 턴에 아래 후보 도시들을 추천했다: "
            f"{candidate_list}. "
            "사용자의 새 메시지가 이 중 한 도시를 골라 관광지·꼭 가봐야 할 곳·맛집을 "
            "묻는 것이면 wants_city_details=true로 하고 selected_city에 정확한 "
            "도시명을 담아라(이름으로 말했든 '1번'처럼 순번으로 말했든 위 목록과 "
            "대조해서 골라라). 예산·기간·목적 등 조건을 바꿔 다시 추천·계획을 "
            "받고 싶어하는 요청이면 wants_city_details=false로 하라."
        ),
    )
    result = intent_agent.invoke({"messages": state["messages"]})
    extracted: FollowupIntent = result["structured_response"]
    intent_data = extracted.model_dump()
    if intent_data.get("selected_city"):
        intent_data["selected_city"] = (
            resolve_city_name(intent_data["selected_city"]) or intent_data["selected_city"]
        )
    return {"special_intent": intent_data}


def route_after_followup_classify(state: AgentState) -> Literal["attractions", "recommend"]:
    """도시 상세 정보를 원하면 관광지·맛집 조회로, 재계획 요청이면 기존 추천 흐름으로 보낸다."""
    intent = state.get("special_intent") or {}
    city = intent.get("selected_city")
    if intent.get("wants_city_details") and city and city in list_supported_cities():
        return "attractions"
    return "recommend"


def attractions_node(state: AgentState) -> dict:
    """사용자가 고른 도시의 관광지·맛집 조회 도구를 호출한다."""
    city = (state.get("special_intent") or {}).get("selected_city")
    result = get_attractions(city=city)
    return {
        "attractions_result": result,
        "trace": state.get("trace", []) + ["get_attractions"],
        "contexts": result["contexts"],
    }


def ask_missing_slot_node(state: AgentState) -> dict:
    """부족한 필수 슬롯을 하나만 되묻는다. 이번 턴엔 도구를 호출하지 않는다."""
    question = _SLOT_QUESTIONS[state["missing_slot"]]
    return {"answer": question, "contexts": [], "trace": []}


def recommend_destinations_node(state: AgentState) -> dict:
    """여행지 추천 도구를 호출한다 (도시를 지정했어도 항상 호출)."""
    slots = state["slots"]
    result = recommend_destinations(
        season=slots["season"], purpose=slots["purpose"], city=slots.get("city")
    )
    return {
        "destinations_result": result,
        "trace": state.get("trace", []) + ["recommend_destinations"],
    }


def route_after_recommend(state: AgentState) -> Literal["plan", "answer"]:
    """지원 범위 밖이면 바로 답변으로, 아니면 일정 플래너로 간다."""
    return "plan" if state["destinations_result"]["in_scope"] else "answer"


def plan_itinerary_node(state: AgentState) -> dict:
    """일정·예산 플래너 도구를 후보 도시마다(최대 _MAX_RECOMMENDATIONS개) 호출한다.

    답변에서 후보를 여러 개 보여주기로 했으므로(generate_answer_node), 대표
    후보 1곳만이 아니라 제시하는 후보 각각의 예산 배분을 계산해 둔다.
    """
    slots = state["slots"]
    candidates = state["destinations_result"]["candidates"][:_MAX_RECOMMENDATIONS]
    results = [
        plan_itinerary(
            city=candidate["city"],
            season=slots["season"],
            nights=slots["nights"],
            total_budget_krw=slots["budget_krw"],
        )
        for candidate in candidates
    ]
    return {"itinerary_results": results, "trace": state["trace"] + ["plan_itinerary"]}


def route_after_plan(state: AgentState) -> Literal["precautions", "answer"]:
    """1순위 후보의 예산이 충분하면 주의사항 조회로, 부족하면 바로 답변으로 간다."""
    return "precautions" if state["itinerary_results"][0]["status"] == "ok" else "answer"


def get_precautions_node(state: AgentState) -> dict:
    """제시할 후보 도시들이 속한 국가마다(중복 없이) 주의사항 조회 도구를 호출한다."""
    candidates = state["destinations_result"]["candidates"][:_MAX_RECOMMENDATIONS]
    countries = list(dict.fromkeys(candidate["country"] for candidate in candidates))
    precautions_by_country = {country: get_precautions(country=country) for country in countries}
    all_contexts = [
        context
        for result in precautions_by_country.values()
        for context in result["contexts"]
    ]
    return {
        "precautions_by_country": precautions_by_country,
        "trace": state["trace"] + ["get_precautions"],
        "contexts": all_contexts,
    }


def generate_answer_node(state: AgentState) -> dict:
    """도구 결과를 근거로 최종 답변을 자연어로 작성한다.

    이 노드의 create_agent는 tools=None이라 도구를 스스로 호출할 권한이
    없고, 상태에 이미 쌓인 사실(도구 결과·contexts)만 사용해 답을 쓴다.
    """
    answer_agent = create_agent(
        model=_llm(),
        tools=None,
        system_prompt=(
            "너는 해외여행 플랜 에이전트다. 아래 도구 실행 결과와 주의사항 "
            "근거(contexts)에 있는 사실만 사용해서 한국어로 답하라. 근거에 "
            "없는 통계나 수치를 만들어내지 마라. 지원 범위 밖이거나 예산이 "
            "부족하면 그 사실을 있는 그대로 안내하라. "
            "총예산을 언급할 땐 슬롯의 budget_krw(원 단위 숫자)를 직접 "
            "만원 단위로 환산하지 말고, 이미 계산된 budget_krw_formatted "
            "문자열을 그대로 써라(직접 환산하면 자릿수를 틀리기 쉽다). "
            "여행지_추천_결과가 in_scope=True이면 candidates 중 최소 2개(있는 "
            "만큼, 최대 3개)를 후보로 함께 제시하라 — 1곳만 언급하지 마라. "
            "requested_city가 있으면 그 도시가 candidates에 있는지 확인해서 "
            "답변에 명시하라: 있으면 맨 먼저 강조하고, 없거나 이번 조건에 "
            "안 맞으면(requested_city_matches가 false) 그 이유(계절·목적 "
            "불일치 등)를 밝히고 나머지 후보 중 최소 2개를 대안으로 제시하라. "
            "일정_예산_결과_후보별(도시별 예산 배분 리스트)이 있으면, 답변에서 "
            "제시하는 후보 도시마다 그 도시(city 값)에 해당하는 예산 배분을 "
            "각각 보여줘라 — 대표 후보 1곳의 금액만 보여주지 마라. "
            "주의사항_결과_국가별(국가명 -> 주의사항)이 있으면, 후보들이 같은 "
            "국가면 그 나라 주의사항을 한 번만 정리해 여러 도시에 공통 적용된다고 "
            "밝히고, 국가가 다른 후보가 섞여 있으면 국가별로 나눠 각각 보여줘라. "
            "(가드레일 상황에서만 쓰이는 일정_예산_결과/주의사항_결과 단수 필드가 "
            "대신 채워져 있으면 그 하나만 사용해서 답하면 된다.) "
            "관광지_맛집_결과가 있으면, 그 도시의 관광지·꼭 가봐야 할 곳·맛집만 "
            "간결히 답하라. has_evidence가 false면 근거 문서가 없다는 사실을 "
            "지어내지 말고 그대로 안내하라."
        ),
    )
    if state.get("attractions_result"):
        # 관광지·맛집 후속 질문일 때는 이전 턴의 추천·예산·주의사항 정보를 아예
        # facts에 안 담는다 — 프롬프트 지시만으로는 반복을 못 막아서(다른 지시와
        # 충돌해 무시되는 걸 확인함), 코드에서 확실히 걸러낸다.
        facts = {
            "슬롯": _slots_for_prompt(state.get("slots") or {}),
            "관광지_맛집_결과": state.get("attractions_result"),
        }
    else:
        facts = {
            "슬롯": _slots_for_prompt(state.get("slots") or {}),
            "여행지_추천_결과": state.get("destinations_result"),
            "일정_예산_결과": state.get("itinerary_result"),
            "일정_예산_결과_후보별": state.get("itinerary_results"),
            "주의사항_결과": state.get("precautions_result"),
            "주의사항_결과_국가별": state.get("precautions_by_country"),
        }
    result = answer_agent.invoke({"messages": [HumanMessage(content=f"도구 실행 결과: {facts}")]})
    final_message: AIMessage = result["messages"][-1]
    return {"answer": final_message.content, "contexts": state.get("contexts", [])}


def build_app():
    """LangGraph 그래프를 구성·컴파일해 반환한다."""
    graph = StateGraph(AgentState)

    graph.add_node("extract_slots", extract_slots_node)
    graph.add_node("ask_missing_slot", ask_missing_slot_node)
    graph.add_node("recommend", recommend_destinations_node)
    graph.add_node("plan", plan_itinerary_node)
    graph.add_node("precautions", get_precautions_node)
    graph.add_node("answer", generate_answer_node)
    graph.add_node("classify_partial_intent", classify_partial_intent_node)
    graph.add_node("scope_check", scope_check_node)
    graph.add_node("precaution_only", precaution_only_node)
    graph.add_node("budget_override_guard", budget_override_guard_node)
    graph.add_node("classify_followup", classify_followup_node)
    graph.add_node("attractions", attractions_node)

    graph.add_edge(START, "extract_slots")
    graph.add_conditional_edges(
        "extract_slots",
        route_after_extract_extended,
        {
            "recommend": "recommend",
            "classify_followup": "classify_followup",
            "scope_check": "scope_check",
            "classify_partial_intent": "classify_partial_intent",
            "ask_missing_slot": "ask_missing_slot",
        },
    )
    graph.add_conditional_edges(
        "classify_partial_intent",
        route_after_classify,
        {
            "scope_check": "scope_check",
            "precaution_only": "precaution_only",
            "budget_override_guard": "budget_override_guard",
            "ask_missing_slot": "ask_missing_slot",
        },
    )
    graph.add_conditional_edges(
        "classify_followup",
        route_after_followup_classify,
        {"attractions": "attractions", "recommend": "recommend"},
    )
    graph.add_edge("ask_missing_slot", END)
    graph.add_edge("scope_check", "answer")
    graph.add_edge("precaution_only", "answer")
    graph.add_edge("budget_override_guard", "answer")
    graph.add_edge("attractions", "answer")
    graph.add_conditional_edges(
        "recommend", route_after_recommend, {"plan": "plan", "answer": "answer"}
    )
    graph.add_conditional_edges(
        "plan", route_after_plan, {"precautions": "precautions", "answer": "answer"}
    )
    graph.add_edge("precautions", "answer")
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=MemorySaver())
