# agent_demo.py
from langchain.agents import create_react_agent, AgentExecutor   # ✅ 新增 AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
import datetime
import os
from dotenv import load_dotenv

load_dotenv()


@tool("get_weather", description="Get today's mock weather for a city.", return_direct=False)
def get_weather(city: str) -> str:
    fake_weather_data = {
        "Tokyo": "晴，25°C",
        "Beijing": "多云，28°C",
        "Akita": "小雨，22°C",
        "Osaka": "晴转多云，27°C",
    }
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f"{today} {city} 天气：{fake_weather_data.get(city, '未找到天气数据')}"


llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

tools = load_tools(["llm-math"], llm=llm)
tools.append(get_weather)

tool_desc = "\n".join(
    f"- {t.name}: {t.description or 'No description'}" for t in tools)
tool_names = ", ".join(t.name for t in tools)

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template).partial(
    tools=tool_desc,
    tool_names=tool_names,
)

# 1) 先创建“规划器” (planner)
planner = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# 2) 用执行器包起来 → 负责跑工具、维护 intermediate_steps、直到 Final Answer
agent = AgentExecutor(
    agent=planner,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# 3) 调用时用 {input: "..."}；Executor 会自动处理 intermediate_steps
resp = agent.invoke({
    "input": "今天东京的天气怎么样？如果 25°C 降温 5 度，还剩多少度？"
})
print("Agent 回答:\n", resp["output"])
