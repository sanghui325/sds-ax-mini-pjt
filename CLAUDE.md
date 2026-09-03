# 프로젝트 규칙

## 기술 스택
- Python, LangChain, LangGraph
- 모델은 Amazon Bedrock (ChatBedrockConverse)
- Agent 생성은 langchain.agents 의 create_agent 를 쓴다

## 폴더 구조 (제출 규약. 바꾸지 않는다)
src/agent.py       메인 에이전트 그래프
src/tools.py       도메인 도구
src/retriever.py   RAG 파이프라인
data/              사용한 문서와 데이터
evaluation/        test_queries.csv 와 평가 리포트

## 주고받는 형식 (제출 규약)
- POST /query 로 받고 question 필드를 읽는다
- 답은 answer, contexts, trace 세 키로 돌려준다
- 정보가 부족해 되물어야 할 때는 question 외 session_id 필드로 이전 대화를 이어간다 (SERVICE.md 4번 정책)

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓴다
- 비밀 값은 .env 에서 읽고 코드에 적지 않는다

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다

## 서비스 규칙 (SERVICE.md 참고)
- 대상 범위: 일본·중국·베트남·태국 12개 도시 (`data/destinations.json` 목록 밖은 지어내지 않고 지원 범위를 안내한다)
- `src/tools.py` 에 둘 도구
  - 여행지 추천: `destinations.json` + `weather.json` 으로 기간(계절)·목적에 맞는 도시 후보를 낸다
  - 일정·예산 플래너: `prices.json` 으로 총예산을 항공권·숙박·교통·식비로 자동 배분한다
  - 주의사항 조회: `visa_safety.md` 를 retriever 로 검색해 근거와 함께 안내한다
- 비자·안전 등 주의사항은 retriever 로 찾은 근거(contexts) 없이 답하지 않는다
- 기간·목적·예산 등 정보가 부족하면 단정하지 않고 되묻는다
- 항공권·숙박·교통·식비 배분 합계는 총예산을 넘지 않는다. 최저가로도 넘으면 배분을 지어내지 않고 예산 부족을 안내한다
- 위 예산 배분 합계 검증은 LLM에 맡기지 않고 일정·예산 플래너 도구 안에서 Python으로 직접 계산·검증한다
- 여행지 추천 → 일정·예산 플래너 → 주의사항 조회 순서는 `src/agent.py`에서 LangGraph 노드를 순차 연결해 보장한다 (에이전트의 자유 도구 선택에 맡기지 않는다)
