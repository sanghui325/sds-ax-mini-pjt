# 해외여행 플랜 Agent

여행 기간·목적·예산만 말하면 여행지 추천부터 일정(항공권·숙박·교통·식비) 배분, 비자·안전 주의사항까지 근거와 함께 한 번에 답해주는 LangGraph 기반 에이전트입니다. 대상은 일본·중국·베트남·태국 12개 도시이며, 필수 정보가 부족하면 임의로 단정하지 않고 되묻고, 근거 없는 답변·예산 초과 배분은 코드 레벨 가드레일로 막습니다. 서비스 요구사항은 [SERVICE.md](SERVICE.md), 프로젝트 규칙은 [CLAUDE.md](CLAUDE.md)를 참고하세요.

## 기술 스택

- Python, LangChain / LangGraph
- 모델: Amazon Bedrock `ChatBedrockConverse` (대화), Bedrock 임베딩 (RAG)
- 벡터스토어: Chroma
- API: FastAPI

## 폴더 구조

```
src/agent.py       메인 에이전트 그래프 (LangGraph StateGraph)
src/tools.py       도메인 도구 (여행지 추천, 일정·예산 플래너, 주의사항 조회)
src/retriever.py   RAG 파이프라인 (Chroma + Bedrock 임베딩)
src/api.py         POST /query FastAPI 진입점
data/              사용한 문서와 데이터 (전부 프로젝트 예시용 더미)
evaluation/        test_queries.csv 와 평가 스크립트·리포트
```

## 준비

1. Python 가상환경을 만들고 의존성을 설치합니다.

   ```bash
   python -m venv .venv
   .venv/Scripts/activate   # Windows
   pip install -r requirements.txt
   ```

2. 프로젝트 루트에 `.env` 파일을 만들고 아래 키를 채웁니다(실제 값은 커밋하지 않습니다).

   ```
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_DEFAULT_REGION=us-east-1
   BEDROCK_MODEL_ID=...            # 예: anthropic.claude 계열 대화 모델
   BEDROCK_EMBEDDING_MODEL_ID=...  # 예: amazon.titan-embed-text-v2:0
   ```

## 실행

```bash
uvicorn src.api:app --reload
```

첫 요청 시 `data/visa_safety.md`를 청크·임베딩해 `data/chroma_db/`에 캐시합니다(문서가 안 바뀌면 재임베딩하지 않습니다).

### API

`POST /query`

| 필드 | 위치 | 설명 |
|---|---|---|
| `question` | 요청 | 사용자 질문(자연어) |
| `session_id` | 요청 (필수) | 대화를 이어갈 세션 식별자. 클라이언트가 새 대화마다 UUID를 생성해서 보냅니다 |
| `answer` | 응답 | 최종 답변 |
| `contexts` | 응답 | 답변 근거로 쓰인 문서 조각 목록 |
| `trace` | 응답 | 이번 턴에 호출된 도구 이름 목록(순서대로) |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "10월 5일부터 9일까지 관광 목적으로 여행 가고 싶어. 예산은 150만원이야.", "session_id": "'$(uuidgen)'"}'
```

필수 정보(기간·목적·예산) 중 빠진 게 있으면 도구를 호출하지 않고 되묻는 질문만 돌려줍니다 — 같은 `session_id`로 다음 턴에 이어서 답하면 됩니다.

## 데이터

- `data/destinations.json`, `data/prices.json`, `data/weather.json`, `data/visa_safety.md` — 4개국(일본·중국·베트남·태국) 12개 도시 기준
- 전부 프로젝트 제출용 더미 데이터입니다. 실제 서비스로 전환하려면 비자·안전 정보는 외교부 해외안전여행(0404.go.kr) 등 공식 출처로, 가격 데이터는 실시간 API로 교체해야 합니다.

## 평가

```bash
python evaluation/run_eval.py             # expected_traits까지 LLM-judge로 참고 판정
python evaluation/run_eval.py --no-llm-judge   # tools/forbidden 자동 채점만 (Bedrock 호출 적음)
```

`evaluation/test_queries.csv`(12건: positive/negative/edge/guardrail)를 순서대로 실행해 `evaluation/eval_report.md`를 생성합니다. `expected_tools`(도구 호출 순서)·`forbidden`(금지 문자열)은 자동 채점되고, `expected_traits`는 참고용으로 답변과 나란히 기록됩니다.
