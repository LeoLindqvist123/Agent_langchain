# math_agent.py

from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import math

load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model="llama3.1",
    headers={"Authorization": f"Bearer {os.getenv('OLLAMA_BEARER_TOKEN')}"},
)

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    except Exception as e:
        return f"Fel: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Användbart för matematiska beräkningar. Input ska vara ett matematiskt uttryck, t.ex. '2 + 2' eller 'sqrt(16)'."
    )
]

system_prompt = """Du är en matematikassistent som hjälper till att lösa beräkningar och matteproblem.
Du svarar alltid på svenska och förklarar uträkningen steg för steg.

Du har tillgång till följande verktyg:
{tools} """

prompt = PromptTemplate.from_template(system_prompt)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": "Vad är 15% av 2500, och vad är kvadratroten av 144?"})
    print("SVAR: ")
    print(result["output"])