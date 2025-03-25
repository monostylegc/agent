from typing import TypedDict, Annotated, List, Literal, Union, cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import asyncio

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import search_pubmed, scrape_with_agent, login

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 환경 변수 로드
load_dotenv()

checkpointer = InMemorySaver()
store = InMemoryStore()

# 상태 정의
class MainState(TypedDict):
    messages: Annotated[MessagesState, add_messages]
    planning_steps: List[str]
    current_step: str
    search_results: List[Document]
    report: str
    review_result: str
    report_result: str

model1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
model2 = ChatOpenAI(model="gpt-4o", temperature=0.1)
model3 = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0.1)


async def main():
    await login()



