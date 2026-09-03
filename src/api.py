"""POST /query 를 처리하는 FastAPI 진입점."""

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agent import build_app

load_dotenv()

app = FastAPI(title="해외여행 플랜 Agent")
_graph = build_app()


class QueryRequest(BaseModel):
    """POST /query 요청 본문."""

    question: str
    session_id: str = Field(
        ...,
        description="대화를 이어갈 세션 식별자. 새 대화라면 클라이언트가 uuid를 직접 생성해 보낸다",
    )


class QueryResponse(BaseModel):
    """POST /query 응답 본문."""

    answer: str
    contexts: list[dict]
    trace: list[str]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """질문을 받아 에이전트 그래프를 실행하고 answer/contexts/trace로 답한다."""
    config = {"configurable": {"thread_id": request.session_id}}
    result = _graph.invoke({"messages": [HumanMessage(content=request.question)]}, config=config)
    return QueryResponse(
        answer=result["answer"],
        contexts=result.get("contexts", []),
        trace=result.get("trace", []),
    )
