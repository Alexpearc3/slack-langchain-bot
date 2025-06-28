import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Gemini model via LangChain wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

@tool
def handle_general_query(input: str) -> str:
    """Answers general-purpose questions outside of Slack or calendar tasks."""
    return llm.invoke(input)

@tool
def reflect_and_complete(input: str = "") -> str:
    """Reflect on previous conversation history via memory and complete unfinished actions."""
    prompt = PromptTemplate.from_template("{input}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": input})