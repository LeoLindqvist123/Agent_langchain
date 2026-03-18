from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model="llama3.1",
    headers={"Authorization": f"Bearer {os.getenv('OLLAMA_BEARER_TOKEN')}"},
)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]

system_prompt = "Du är en hjälpsam agent som kan svara på alla frågor som finns på wikipedia"

prompt = PromptTemplate.from_template(system_prompt)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    result = agent_executor.invoke({
        "input": "När uppfann man macbook"
    })
    print("SVAR: ")
    print(result["output"])