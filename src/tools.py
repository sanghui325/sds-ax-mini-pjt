"""여행지 추천, 일정·예산 배분, 주의사항 조회를 담당하는 도메인 도구."""

import json
from functools import lru_cache
from pathlib import Path

from src.retriever import retrieve

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TIERS = ("고", "중", "저")


@lru_cache(maxsize=None)
def _load_json(filename: str) -> tuple[dict, ...]:
    """data/ 아래 JSON 파일을 한 번만 읽어 캐시한다."""
    path = _DATA_DIR / filename
    with path.open(encoding="utf-8") as file:
        return tuple(json.load(file))


def _destinations() -> tuple[dict, ...]:
    """data/destinations.json 내용을 반환한다."""
    return _load_json("destinations.json")


def _prices() -> tuple[dict, ...]:
    """data/prices.json 내용을 반환한다."""
    return _load_json("prices.json")


def _supported_scope_summary() -> list[dict]:
    """지원 중인 국가·도시 목록을 안내용으로 요약한다."""
    return [{"country": row["country"], "city": row["city"]} for row in _destinations()]


def list_supported_cities() -> list[str]:
    """지원 중인 12개 도시명 목록만 반환한다."""
    return [row["city"] for row in _destinations()]


def resolve_city_name(place: str) -> str | None:
    """언급된 지명 문자열 안에서 지원 중인 도시명을 찾아 정식 이름으로 반환한다.

    "베트남 하노이"처럼 국가명이 붙어 있어도 포함된 도시명("하노이")을 찾아내기
    위한 것이다 — LLM이 지명을 추출할 때 완전히 깔끔한 도시명만 주지 않을 수
    있어서, 완전일치 대신 포함 관계로 매칭한다. 찾지 못하면 None을 반환한다.
    """
    for city in list_supported_cities():
        if city in place:
            return city
    return None


def recommend_destinations(season: str, purpose: list[str], city: str | None = None) -> dict:
    """기간(계절)과 목적에 맞는 여행지 후보를 추천한다.

    도시를 이미 지정한 경우에도 이 함수는 항상 전체 후보를 계산한다 —
    지정한 도시가 조건에 맞는지 확인하고, 필요하면 대안도 함께 보여주기 위해서다.
    지원 범위(4개국 12개 도시) 밖의 도시이거나 조건에 맞는 후보가 하나도
    없으면 in_scope=False로 알리고 지어낸 후보를 만들지 않는다.
    """
    supported_cities = {row["city"] for row in _destinations()}
    out_of_scope_city = city is not None and city not in supported_cities

    candidates = [
        dict(row)
        for row in _destinations()
        if season in row["season_recommend"] and set(purpose) & set(row["purpose"])
    ]

    if out_of_scope_city or not candidates:
        return {
            "in_scope": False,
            "candidates": [],
            "requested_city": city,
            "requested_city_matches": False if city else None,
            "supported_scope": _supported_scope_summary(),
        }

    requested_city_matches = None
    if city is not None:
        requested_city_matches = any(row["city"] == city for row in candidates)
        if requested_city_matches:
            matched = [row for row in candidates if row["city"] == city]
            rest = [row for row in candidates if row["city"] != city]
            candidates = matched + rest

    return {
        "in_scope": True,
        "candidates": candidates,
        "requested_city": city,
        "requested_city_matches": requested_city_matches,
        "supported_scope": _supported_scope_summary(),
    }


def _find_price_row(city: str, season: str) -> dict | None:
    """도시·계절에 해당하는 가격 데이터 한 행을 찾는다."""
    for row in _prices():
        if row["city"] == city and row["season"] == season:
            return row
    return None


def plan_itinerary(city: str, season: str, nights: int, total_budget_krw: int) -> dict:
    """총예산을 항공권·숙박·교통·식비로 자동 배분한다.

    배분 합계가 총예산을 넘지 않는지 이 함수 안에서 직접 계산·검증한다
    (LLM에 맡기지 않는다). 숙박 등급을 고→중→저 순으로 낮춰가며 예산 안에
    들어오는 가장 좋은 등급을 고르고, 최저가(저 등급)로도 예산을 넘으면
    배분을 지어내지 않고 부족액만 알린다.
    """
    price_row = _find_price_row(city, season)
    if price_row is None:
        return {"status": "no_price_data", "city": city, "season": season}

    days = nights + 1  # 숙박은 nights박, 교통·식비는 days일 기준으로 계산
    flight_cost = price_row["flight_price_krw"]["min"]
    transport_cost = days * price_row["local_transport_krw_per_day"]
    food_cost = days * price_row["food_price_krw_per_day"]

    for tier in _TIERS:
        hotel_cost = nights * price_row["hotel_price_krw_per_night"][tier]
        total_used = flight_cost + hotel_cost + transport_cost + food_cost
        if total_used <= total_budget_krw:
            return {
                "status": "ok",
                "city": city,
                "season": season,
                "nights": nights,
                "days": days,
                "tier_selected": tier,
                "breakdown": {
                    "항공권": flight_cost,
                    "숙박": hotel_cost,
                    "교통": transport_cost,
                    "식비": food_cost,
                },
                "total_used_krw": total_used,
                "total_budget_krw": total_budget_krw,
                "shortfall_krw": 0,
            }

    lowest_tier = _TIERS[-1]
    hotel_cost = nights * price_row["hotel_price_krw_per_night"][lowest_tier]
    minimum_required = flight_cost + hotel_cost + transport_cost + food_cost
    return {
        "status": "insufficient_budget",
        "city": city,
        "season": season,
        "nights": nights,
        "days": days,
        "tier_selected": None,
        "breakdown": None,
        "total_used_krw": None,
        "total_budget_krw": total_budget_krw,
        "shortfall_krw": minimum_required - total_budget_krw,
    }


def get_precautions(country: str, focus: str | None = None) -> dict:
    """국가별 비자·안전 주의사항을 retriever로 검색해 근거와 함께 반환한다.

    근거 문서를 찾지 못하면 has_evidence=False를 반환한다 — 상위 답변
    생성 단계는 이 경우 내용을 지어내지 말고 근거 없음으로 안내해야 한다.
    """
    query = f"{country} {focus}".strip() if focus else f"{country} 비자 안전 주의사항"
    contexts = retrieve(query=query, country=country)
    return {"has_evidence": bool(contexts), "contexts": contexts, "country": country}
