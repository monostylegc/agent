from typing import TypedDict, Annotated, List, Literal, Union, cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import asyncio
from datetime import datetime

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import search_pubmed, scrape_with_agent, login
# 프롬프트 모듈 임포트
from prompts import PLANNER_PROMPT, WRITER_PROMPT, REVIEWER_PROMPT

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor
from langgraph_swarm import create_handoff_tool, create_swarm
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

# 환경 변수 로드
load_dotenv()

# 디버그 모드 설정
VERBOSE = True

# 콘솔 출력 설정
console = Console()

# 체크포인터 및 저장소 설정
checkpointer = InMemorySaver()
store = InMemoryStore()

# 추가 도구
@tool
def get_current_datetime() -> str:
    """현재 날짜와 시간을 반환합니다."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def post_report(report_content: str) -> str:
    """보고서 내용을 저장하고 출력하는 도구."""
    # 보고서 저장
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # 보고서 출력
    console.print(Markdown(report_content))
    
    return "보고서가 성공적으로 저장 및 출력되었습니다."

mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.1)
claude = ChatAnthropic(model="claude-3.7-sonnet-latest", temperature=0.1)
o3mini = ChatOpenAI(model="o3-mini", temperature=0.1)
# 여러 모델 설정
planner_model = gpt4o
writer_model = gpt4o
reviewer_model = gpt4o

# 플래너 에이전트 생성
planner_agent = create_react_agent(
    model=planner_model,
    name="PlannerAgent",
    checkpointer=checkpointer,
    tools=[
        search_pubmed,
        create_handoff_tool(
            agent_name="WriterAgent",
            description="연구 계획을 WriterAgent에게 전달하여 계획에 따른 연구 내용을 작성하도록 합니다."
        ),
    ],
    prompt=PLANNER_PROMPT,  # 기본 프롬프트 사용
)

# 작성자 에이전트 생성
writer_agent = create_react_agent(
    model=writer_model,
    name="WriterAgent",
    checkpointer=checkpointer,
    tools=[
        scrape_with_agent, 
        search_pubmed,
        create_handoff_tool(
            agent_name="PlannerAgent",
            description="연구 계획 수정이나 추가 정보 수집이 필요한 경우 PlannerAgent에게 작업을 인계합니다."
        ),
        create_handoff_tool(
            agent_name="ReviewerAgent",
            description="작성된 연구 내용을 ReviewerAgent에게 전달하여 검토하도록 합니다."
        ),
    ],
    prompt=WRITER_PROMPT,  # 기본 프롬프트 사용
)

# 검토자 에이전트 응답 형식
class ReviewerResponse(BaseModel):
    review: str = Field(description="검토 내용과 피드백")
    accept: bool = Field(description="내용 수용 여부 (True: 수용, False: 거부)")

# 검토자 에이전트 생성
reviewer_agent = create_react_agent(
    model=reviewer_model,
    name="ReviewerAgent",
    checkpointer=checkpointer,
    tools=[
        create_handoff_tool(
            agent_name="WriterAgent",
            description="검토 결과를 바탕으로 WriterAgent에게 작업을 다시 전달하여 내용을 수정하도록 합니다."
        ),
        create_handoff_tool(
            agent_name="PlannerAgent",
            description="연구 계획의 전반적인 수정이 필요한 경우 PlannerAgent에게 작업을 인계합니다."
        ),
        post_report
    ],
    prompt=REVIEWER_PROMPT,  # 기본 프롬프트 사용
    response_format=ReviewerResponse
)

# 스웜 생성 및 구성
research_swarm = create_swarm(
    [
        planner_agent,
        writer_agent,
        reviewer_agent,
    ],
    default_active_agent="PlannerAgent",
).compile(
    name="ResearchSwarm",
    checkpointer=checkpointer,
)

async def main():
    await login()
    
    # 연구 주제 설정
    research_topic = """Vacum disc의 증상과 감별진단의 방법과 최신 치료방법에 대한 연구"""
    
    console.print("[bold blue]연구 프로세스 시작[/bold blue]")
    
    # 스웜 실행
    async for response in research_swarm.astream(
        {"messages": [{"role": "user", "content": research_topic}]},
        {
            "configurable": {
                "thread_id": "thread-1",
                "recursion_limit": 1000,
            }
        },
        stream_mode="updates",
    ):
        # 에이전트 정보 추출
        for agent_name, agent_data in response.items():
            # 에이전트 노드 이름 출력
            print(f"[Node: {agent_name}]")
            print()
            
            # 메시지 처리
            if 'messages' in agent_data and agent_data['messages']:
                latest_message = agent_data['messages'][-1]
                
                # AI 메시지 출력
                if hasattr(latest_message, 'name') and latest_message.name == agent_name:
                    print("================================== Ai Message ==================================")
                    # 툴 콜이 있는 경우
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        print("Tool Calls:")
                        for tool_call in latest_message.tool_calls:
                            print(f"  {tool_call.get('name', 'Unknown')} ({tool_call.get('id', 'Unknown')})")
                    
                    # 메시지 내용 출력
                    print(latest_message.content)
                    
                # 툴 메시지 출력
                elif hasattr(latest_message, 'name') and latest_message.name != agent_name:
                    print("================================= Tool Message =================================")
                    print(f"Name: {latest_message.name}")
                    print()
                    print(latest_message.content)
                    
            print() # 줄바꿈으로 구분

if __name__ == "__main__":
    asyncio.run(main())



