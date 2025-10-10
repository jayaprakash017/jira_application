import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

async def interactive_jira_agent():
    print("Starting interactive Jira agent with MCP...")
    print("Type 'quit' or 'exit' to stop.\n")

    # Initialize MCP client
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
        tool_names = [tool.name for tool in tools]
        #print(f"Available tools: {tool_names}\n")


        # Initialize LLM and agent
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.3  
        )
        agent = create_react_agent(llm, tools=tools)

        # Start conversation loop
        messages = []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit", "bye"}:
                print("Goodbye!")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            try:
                response = await agent.ainvoke({"messages": messages})
                ai_message = response["messages"][-1].content
                print(f"Agent: {ai_message}\n")

                # Append AI response to history for context
                messages = response["messages"]  

            except Exception as e:
                print(f"Error during agent execution: {e}\n")
                

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        try:
            await client.close()
            print("MCP client closed.")
        except Exception as e:
            print(f"Error closing client: {e}")

async def main():
    await interactive_jira_agent()

if __name__ == "__main__":
    asyncio.run(main())