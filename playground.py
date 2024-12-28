from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.openai import OpenAIChat
import openai
import os
import phi.api
import phi
from phi.playground import Playground, serve_playground_app
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

web_search_agent=Agent(
    name="web_search_agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the response."],
    show_tools_calls=True,
    markdown=True,  
)


# financial agent
financial_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data."],
    show_tools_calls=True,
    markdown=True,

)

app=Playground(agents=[web_search_agent, financial_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app",reload=True)