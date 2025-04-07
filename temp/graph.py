from typing import TypedDict, Annotated, List, Literal, Union, cast
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import asyncio
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from tools import search_pubmed, scrape_with_agent, login

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

# 환경 변수 로드
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class PubMedResult(BaseModel):
    title: str
    url: str
    abstract: str
    authors: List[str]

class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: list[str]
    papers: list[PubMedResult]
    selected_papers: list[PubMedResult]
    post : str
    review : str
    attempt_count: int

def start_with_user_input(state: MainState) -> Command:
    user_input = input("질문을 입력해주세요: ")
    return Command(
        update={"messages": HumanMessage(content=user_input), "attempt_count": 1},
        goto="query_generation"
    )

def query_generation(state: MainState) -> Command:
    system_message = SystemMessage(
        content="""
        당신은 PubMed 검색을 위한 최적화된 쿼리를 생성하는 전문가입니다. 사용자의 의학 연구 질문이나 토픽을 PubMed에서 효과적으로 검색할 수 있는 쿼리로 변환해주세요.
        """
    )
    msg = state["messages"]

    class QueryGenerationOutput(BaseModel):
        querys : List[str] = Field(description="검색 쿼리 목록")
    
    model_with_structured_output = model.with_structured_output(
        QueryGenerationOutput,
        method="function_calling"
    )
    messages = [system_message] + msg
    response = model_with_structured_output.invoke(messages)
    
    # 구조화된 출력을 문자열로 변환하여 AIMessage에 넣기
    response_str = f"생성된 검색 쿼리:\n" + "\n".join(response.querys)
      
    return Command(
        update={"messages": AIMessage(content=response_str), "query": response.querys},
        goto="pubmed_search"
    )

def pubmed_search(state: MainState) -> Command:
    papers = []
    
    for query in state["query"]:
        response = search_pubmed.invoke(query)
        papers.extend(response)
    
    return Command(
        update={"messages": AIMessage(
            content=f"""검색된 논문은 다음과 같습니다.\n
            {papers}"""),
        "papers": papers
        },
        goto="paper_selection"
    )

def paper_selection(state: MainState) -> Command:
    #검색결과가 모자라면 query_generation 노드로 돌아가서 질문을 수정해야 함
    # tools.py에서 정의한 PubMedResult를 활용
    
    # 시도 횟수 확인 (없으면 1로 초기화)
    attempt_count = state.get("attempt_count", 1)
    
    # 시도 횟수가 3회 이상이면 선택된 논문이 적어도 진행
    if attempt_count >= 3:
        return Command(
            update={
                "messages": AIMessage(content="충분한 논문을 찾지 못했지만, 3회 시도 후 진행합니다."),
                "selected_papers": state["papers"][:5] if len(state["papers"]) >= 5 else state["papers"]
            },
            goto="post_write"
        )
    
    class PaperSelectionOutput(BaseModel):
        selected_papers: list[PubMedResult] = Field(description="선택된 논문 목록")
        need_more_papers: bool = Field(description="추가 논문 검색이 필요한지 여부")
        query_suggestion: str = Field(description="검색 쿼리 개선 제안")
    
    system_message = SystemMessage(
        content="""
        당신은 의학 연구 논문을 평가하고 선별하는 전문가입니다. 제공된 PubMed 검색 결과에서 사용자의 연구 질문에 가장 적합한 논문들을 선택해주세요.
        
        논문 선택 시 다음 기준을 고려하세요(중요도 순):
        
        1. 연관성 (가장 중요한 기준):
           - 연구 질문/토픽과의 직접적인 관련성이 가장 중요합니다
           - 핵심 개념이나 중재가 제목이나 초록에 명확히 언급되어야 함
           - 연구 질문의 모든 주요 요소(질병, 치료법, 결과 등)를 다루는지 확인
           - 초록에서 사용자 질문에 대한 직접적인 답변이나 관련 정보를 제공하는지 확인
           - 연관성이 낮은 논문은 연구 설계나 품질이 우수하더라도 선택하지 마세요
        
        2. 연구 품질 (2차적 고려사항):
           - 연구 설계의 강점(RCT > 코호트 > 횡단 연구 > 사례 보고 등)
           - 표본 크기의 적절성
           - 최신성(가능한 최근 5년 이내 발표된 연구 우선)
        
        3. 영향력 (3차적 고려사항):
           - 저널의 영향력 지수(Impact Factor)
           - 저자의 전문성 및 인용 지수
           - 타 연구에서의 인용 횟수
        
        4. 다양성 (보완적 고려사항):
           - 다양한 관점을 제공하는 논문들 포함
           - 상반된 결과를 보고하는 연구도 포함 고려
           - 체계적 문헌고찰이나 메타분석이 있다면 우선 선택
        
        연관성 평가 방법:
        - 사용자 질문의 핵심 키워드를 모두 추출하세요
        - 각 논문의 제목과 초록에서 이 키워드들의 출현 빈도와 중요도를 확인하세요
        - 초록이 연구 질문에 직접적으로 답변하는지 평가하세요
        - 논문의 주요 주제와 연구 질문의 일치도를 0-10점으로 평가하고, 7점 이상인 논문만 고려하세요
        
        선택 과정:
        1. 각 논문의
           - 제목
           - 초록
           - 저자
           - 게재 정보
           를 검토하세요.
        
        2. 각 논문이 사용자 질문에 얼마나 관련되어 있는지 먼저 평가하고, 관련성이 낮은 논문은 바로 제외하세요.
        
        3. 관련성이 높은 논문 중에서 연구 품질, 영향력, 다양성을 고려하여 5편 이상을 최종 선택하세요.
           (연관성 점수 7점 이상인 논문이 5편 미만이라면, 이를 표시하고 검색 쿼리 개선 제안을 해주세요)
        
        4. 선택한 논문마다 왜 그 논문이 중요한지 간략한 이유를 제시하되, 연관성에 관한 설명을 가장 먼저 언급하세요.
        
        출력 형식:
        1. 선택한 논문 목록(제목과 URL 포함)
        2. 각 논문별 선택 이유 간략 설명(연관성 관련 이유를 먼저 언급)
        3. 선택한 논문들이 함께 사용자 질문에 어떻게 종합적인 답변을 제공할 수 있는지 설명
        4. 연관성 높은 논문이 5편 미만인 경우, 검색 쿼리 개선을 위한 제안
        
        참고: 제공된 papers 리스트에서 각 논문의 title, url, abstract, authors 정보를 확인하여 평가하세요.
        """
    )
    
    # 사용자 질문과 검색된 논문 목록을 함께 모델에 전달
    human_message = HumanMessage(content=f"""
    사용자 질문: {state["messages"][0].content}
    
    검색된 논문 목록:
    {state["papers"]}
    
    사용자 질문과 연관성이 높은 논문을 선택해주세요. 연관성이 낮은 논문은 반드시 제외해야 합니다.
    연관성이 7점 이상(10점 만점)인 논문을 선택하되, 최소 5편 이상 선택하는 것이 목표입니다.
    만약 연관성 높은 논문이 5편 미만이라면, need_more_papers를 True로 설정하고, 
    어떻게 검색 쿼리를 개선하면 더 관련성 높은 논문을 찾을 수 있을지 query_suggestion에 제안해주세요.
    각 논문에 대해 연관성 점수와 그 이유를 간략히 설명해주세요.
    """)
    
    model_with_structured_output = model.with_structured_output(
        PaperSelectionOutput,
        method="function_calling"
    )
    # 메시지 리스트로 전달
    messages = [system_message, human_message]
    response = model_with_structured_output.invoke(messages)
    
    # 선택된 논문이 5편 미만이면 query_generation으로 돌아감
    if response.need_more_papers or len(response.selected_papers) < 5:
        # 시도 횟수 증가 및 업데이트
        attempt_count += 1
        
        return Command(
            update={"messages": AIMessage(content=f"""
            연관성 높은 논문이 충분하지 않습니다. 검색 쿼리를 개선하여 다시 시도하겠습니다. (시도 {attempt_count}/3)
            
            제안된 검색 쿼리 개선 방향: {response.query_suggestion}
            """),
            "attempt_count": attempt_count  # 시도 횟수 업데이트
            },
            goto="query_generation"
        )
    else:
        return Command(
            update={"messages": AIMessage(content=f"사용자 질문과 가장 연관성이 높은 논문 {len(response.selected_papers)}편을 선택했습니다."), 
                    "selected_papers": response.selected_papers},
            goto="post_write"
        )
    
# async def scrape_paper(state: MainState) -> MainState:
#     pass

def post_write(state: MainState) -> Command:
    system_message = SystemMessage(
        content="""
        당신은 의학 연구 논문을 작성하는 전문가입니다. 선택된 논문들의 내용을 바탕으로 의학 논문 형식에 맞는 보고서를 작성해주세요.
        
        다음과 같은 의학 논문의 표준 구조를 따라 작성하세요:
        
        1. 제목(Title): 연구 주제를 명확히 반영하는 간결하고 구체적인 제목
        
        2. 초록(Abstract): 200-300단어 내외로 연구의 목적, 방법, 결과, 결론을 요약
           - 배경(Background)
           - 목적(Objective)
           - 방법(Methods)
           - 결과(Results)
           - 결론(Conclusion)
        
        3. 서론(Introduction): 연구 배경, 문제 제기, 선행 연구 검토, 연구 목적 및 가설 제시
           - 연구 주제의 중요성과 배경
           - 현재까지의 연구 동향과 한계점
           - 본 연구의 목적과 의의
        
        4. 방법(Methods): 연구 설계, 대상, 자료 수집 및 분석 방법 상세 기술
           - 연구 설계(연구 유형, 기간)
           - 연구 대상 및 표본 선정 기준
           - 자료 수집 방법
           - 분석 방법 및 통계 기법
        
        5. 결과(Results): 주요 발견과 분석 결과를 객관적으로 제시
           - 주요 결과를 논리적 순서로 제시
           - 표, 그래프 등의 데이터 요약 (필요시)
           - 통계적 유의성
        
        6. 고찰(Discussion): 결과의 해석, 기존 연구와의 비교, 의의 및 한계점
           - 주요 발견에 대한 해석
           - 선행 연구와의 일치점 및 차이점
           - 연구의 강점과 한계점
           - 임상적 의의
        
        7. 결론(Conclusion): 연구의 핵심 발견과 시사점, 향후 연구 방향 제안
        
        8. 참고문헌(References): 선택된 논문들을 Vancouver 스타일로 인용
        
        작성 시 주의사항:
        - 객관적이고 과학적인 문체를 사용할 것
        - 선택된 논문들의 핵심 내용과 증거를 정확하게 인용할 것
        - 논문들 간의 상충되는 결과나 관점이 있다면 균형 있게 제시할 것
        - 의학 용어와 약어를 적절히 사용하되, 첫 등장 시 전체 용어를 함께 표기할 것
        - 선택된 논문들의 통합적 분석을 통해 새로운 통찰을 제공할 것
        """
    )
    
    # 사용자 질문과 선택된 논문들을 함께 모델에 전달
    papers_info = []
    for i, paper in enumerate(state["selected_papers"]):
        papers_info.append(f"""
        논문 {i+1}:
        제목: {paper.title}
        저자: {', '.join(paper.authors)}
        초록: {paper.abstract}
        URL: {paper.url}
        """)
    
    human_message = HumanMessage(content=f"""
    연구 질문: {state["messages"][0].content}
    
    선택된 논문 정보:
    {''.join(papers_info)}
    
    위 논문들의 내용을 바탕으로 의학 논문 형식에 맞는 보고서를 작성해주세요.
    """)
    
    # 보고서 생성 - 메시지 리스트로 전달
    messages = [system_message, human_message]
    report_response = model.invoke(messages)
    
    # 생성된 보고서 저장
    return Command(
        update={
            "messages": AIMessage(content="의학 논문 형식의 보고서가 작성되었습니다."),
            "post": report_response.content
        },
        goto="review_node"
    )

def review_node(state: MainState) -> Command:
    system_message = SystemMessage(
        content="""
        당신은 의학 연구 논문을 검토하고 편집하는 전문 편집자입니다. 작성된 보고서를 검토하고 직접 수정하여 개선된 최종 버전을 제공해주세요.
        
        다음 측면에서 논문을 평가하고 직접 수정하세요:
        
        1. 구조적 완성도:
           - 의학 논문의 표준 구조(제목, 초록, 서론, 방법, 결과, 고찰, 결론, 참고문헌)를 확인
           - 누락된 섹션이 있다면 추가
           - 각 섹션의 길이와 내용 균형 조정
        
        2. 내용의 정확성:
           - 인용된 논문의 내용을 정확하게 반영하도록 수정
           - 사실 관계 오류 수정
           - 모호하거나 불명확한 표현 명확화
        
        3. 논리적 일관성:
           - 주장과 증거 간의 연결이 논리적이도록 재구성
           - 논리적 비약이나 모순 해결
           - 서론에서 결론까지 일관된 흐름 유지
        
        4. 학술적 표현:
           - 의학 용어와 표현을 정확하게 수정
           - 약어 사용 표준화 (첫 등장 시 전체 용어 명시)
           - 학술적 문체로 일관되게 수정
        
        5. 참고문헌 인용:
           - Vancouver 스타일에 맞게 인용 형식 수정
           - 인용 순서와 번호 체계 확인
           - 누락된 인용 추가
        
        6. 언어 및 문법:
           - 문법, 철자, 구두점 오류 수정
           - 문장 구조 개선
           - 불필요한 반복 제거
        
        수정 방식:
        - 원본 보고서의 전체적인 구조와 내용을 유지하면서 수정하세요
        - 수정된 보고서 전체를 완성된 형태로 제공하세요
        - 논문의 각 섹션에 대해 충실하게 내용을 보완하고 수정하세요
        
        출력 형식:
        1. 간략한 수정 요약 (수정한 주요 사항 3-5가지 나열)
        2. 수정된 보고서 전문 (모든 섹션 포함)
        """
    )
    
    human_message = HumanMessage(content=f"""
    다음 의학 연구 보고서를 검토하고 직접 수정해주세요. 개선점을 단순히 제안하는 것이 아니라, 직접 편집된 최종 버전을 작성해주세요:
    
    {state["post"]}
    """)
    
    # 리뷰 및 수정 생성 - 메시지 리스트로 전달
    messages = [system_message, human_message]
    review_response = model.invoke(messages)
    
    return Command(
        update={
            "messages": AIMessage(content="보고서 검토 및 수정이 완료되었습니다."),
            "review": review_response.content,
            "post": review_response.content.split("수정된 보고서 전문:", 1)[-1].strip() if "수정된 보고서 전문:" in review_response.content else review_response.content
        },
        goto="post_publish"
    )

def post_publish(state: MainState) -> Command:
    # 최종 보고서와 리뷰 결과를 합쳐서 출력
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    console = Console()
    console.print(f"\n[bold]===== 의학 연구 보고서 ({now}) =====[/bold]\n")
    console.print(Markdown(state["post"]))
    
    console.print("\n[bold]===== 편집자 수정 요약 =====[/bold]\n")
    
    # 수정 요약만 추출
    review_summary = state["review"].split("수정된 보고서 전문:", 1)[0].strip() if "수정된 보고서 전문:" in state["review"] else state["review"]
    console.print(Markdown(review_summary))
    
    # 파일로 저장
    report_filename = f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"# 의학 연구 보고서 ({now})\n\n")
        f.write(state["post"])
        f.write("\n\n## 편집자 수정 요약\n\n")
        f.write(review_summary)
    
    return Command(
        update={
            "messages": AIMessage(content=f"최종 수정된 보고서가 '{report_filename}' 파일로 저장되었습니다.")
        },
        goto=END
    )

graph = StateGraph(MainState)
graph.add_node("start", start_with_user_input)
graph.add_node("query_generation", query_generation)
graph.add_node("pubmed_search", pubmed_search)
graph.add_node("paper_selection", paper_selection)
graph.add_node("post_write", post_write)
graph.add_node("review_node", review_node)
graph.add_node("post_publish", post_publish)

graph.set_entry_point("start")

checkpointer = InMemorySaver()

# 실행 가능한 애플리케이션으로 컴파일
app = graph.compile(checkpointer=checkpointer)

# 메인 실행 코드
async def run_test():
    # 먼저 로그인 진행
    # print("인제대학교 메드라인에 로그인 중...")
    # await login(headless=False)
    # print("로그인 완료. 세션이 준비되었습니다.")
    
    # 그래프 실행
    print("\n의학 연구 논문 작성 에이전트를 시작합니다.")
    print("========================================\n")
    
    inputs = {"messages": []}
    for chunk in app.stream(inputs, stream_mode="updates",config={"thread_id": "thread-1","recursion_limit":1000}):
       print(chunk)
    
    print("\n========================================")
    print("프로세스가 완료되었습니다.")
    
if __name__ == "__main__":
    asyncio.run(run_test())


