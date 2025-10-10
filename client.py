import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

async def test_via_boards():
    print("Listing JIRA Projects via Agile Boards...")
    client = MultiServerMCPClient({
        "mcp-atlassian": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e", "JIRA_URL",
                "-e", "JIRA_USERNAME", 
                "-e", "JIRA_API_TOKEN",
                "mcp/atlassian"
            ],
            "env": {
                "JIRA_URL": "https://bhanuprakashch.atlassian.net",
                "JIRA_USERNAME": "bhanuprakashchegondi@gmail.com",
                "JIRA_API_TOKEN": "ATATT3xFfGF06zMW1R4KCD5XTo3OaT4U1llCVruDywzZkMl6b73vbuvnv6RXF41dJMu_9BlkRwY1E-WKfYrlE2j2CzVGA4OSKceJlsNdaCMiJF2yalXo5SkMahlgvKqLgUryWiwbHFOxPJA-kIqsTQ2HqFPtlm6CVgpARcWn0XyhxDNTUmlIbQc=0CA1984D"
            },
            "transport": "stdio"
        }
    })
    
    try:
        tools = await client.get_tools()
        print(f"Available tools: {[tool.name for tool in tools]}\n")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0
        )

        agent = create_react_agent(llm, tools=tools)

        # Use agile boards to find projects
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": """List all tickets assigned to me, including their summaries, current statuses and priority."""}]}
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