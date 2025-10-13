import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL")


async def test_via_boards():
    print("Listing JIRA Projects via Agile Boards...")
    client = MultiServerMCPClient({
        "mcp-atlassian": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "--env-file",
                "/home/bhanu/Documents/jira_mcp/.env",
                "mcp/atlassian"
            ],
            "transport": "stdio"
        }
    })
    
    try:
        tools = await client.get_tools()
        print(f"Available tools: {[tool.name for tool in tools]}\n")


        llm = ChatOpenAI(model=OPEN_AI_MODEL,
                        temperature=OPENAI_TEMPERATURE,
                        api_key=OPENAI_API_KEY,
                        base_url=INFERENCE_SERVER_URL

        )

        agent = create_react_agent(llm, tools=tools)

        response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": """list all the tickets assigned to me including their current status and priorities."""}]}
)

        print("\n" + "="*50)
        print("Response:", response["messages"][-1].content)
        print("="*50)
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        try:
            await client.close()
        except:
            pass

async def main():
    result = await test_via_boards()
    print(f"\nResult: {'working' if result else 'not working'}")

if __name__ == "__main__":
    asyncio.run(main())