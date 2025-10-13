import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve configuration from environment variables
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL")


async def run_jira_assistant():
    """
    Main function to run the interactive JIRA assistant.
    Creates an agent with JIRA tools and allows interactive chat.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("Starting JIRA Assistant...")
    
    # Check if required environment variables are set
    required_vars = ["OPEN_AI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        return False

    # Initialize MCP client with JIRA configuration
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
        print("Connecting to MCP client...")
        # Get available tools from the MCP client
        tools = await client.get_tools()
        available_tools = [tool.name for tool in tools]
        print(f"Available tools: {available_tools}\n")

        # Check if any tools are available
        if not tools:
            print("No tools available from MCP client")
            return False

        print("Initializing LLM...")
        # Create ChatOpenAI instance with configuration
        llm = ChatOpenAI(
            model=OPEN_AI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            base_url=INFERENCE_SERVER_URL
        )

        print("Creating JIRA assistant agent with memory...")
        
        # STRICT system prompt for the JIRA assistant
        system_prompt = """You are a comprehensive JIRA assistant with access to 27 different JIRA tools.

AVAILABLE TOOLS:
- User Management: Get user profiles
- Issue Operations: Search, create, update, delete, transition issues
- Issue Details: Get details, comments, worklogs, attachments, changelogs
- Project Operations: Get project issues, fix versions
- Board & Sprint Management: Get boards, sprints, board issues, create/update sprints
- Relationships: Get/create/remove issue links, link to epics
- Time Tracking: Add worklogs, get worklogs

GENERAL WORKFLOW PRINCIPLES:

1. FOR ALL REQUESTS:
   - Use the most appropriate tool for the task
   - Ask for clarification if parameters are unclear
   - Provide clear, actionable responses

2. SPECIFIC WORKFLOW FOR ISSUE CREATION:
   When asked to create an issue:
   a) First search for available projects using: "project IS NOT NULL ORDER BY created DESC" limit=20
   b) Show the user available project keys from search results
   c) Ask user to select a project key from the list
   d) Collect required parameters: summary, issue_type
   e) Optionally collect: description, assignee, components
   f) Only create the issue after all required information is confirmed

3. PARAMETER HANDLING:
   - For searches: Always use reasonable limits (1-50)
   - For issue keys: Ask if not provided
   - For project keys: Discover available ones before creation
   - For required fields: Never proceed without them

4. SMART TOOL SELECTION:
   - Use jira_search_issues_jql for flexible searching
   - Use jira_get_project_issues for project-specific queries
   - Use jira_get_issue_details for single issue deep-dives
   - Use appropriate tools based on what user asks for

5. ERROR PREVENTION:
   - Validate required parameters before tool execution
   - Provide helpful error messages if things fail
   - Guide users through proper workflows

6. USER INTERACTION:
   - Be conversational and helpful
   - Explain what you're doing and why
   - Confirm actions before executing destructive operations (delete, etc.)
   - Provide summaries of results

EXAMPLES OF PROPER INTERACTIONS:

For "create a bug ticket":
1. Search for projects
2. Show available project keys
3. Ask user to select one
4. Ask for summary and description
5. Create the issue

For "show my assigned tickets":
1. Use jira_search_issues_jql with: "assignee = currentUser() ORDER BY created DESC" limit=20

For "get issue details for PROJ-123":
1. Use jira_get_issue_details with issue_key="PROJ-123"

For "add comment to PROJ-456":
1. Ask for comment content
2. Use jira_add_comment_to_issue

Remember: Be helpful, use the right tools, and guide users through proper workflows."""

        # Create memory saver for persistent conversation history
        memory = MemorySaver()
        
        # Create agent with memory
        agent = create_react_agent(
            model=llm, 
            tools=tools,
            checkpointer=memory
        )
        
        # Configuration for the agent with thread ID for memory persistence
        config = {"configurable": {"thread_id": "jira_assistant_session"}}

        print("JIRA Assistant is ready! Type 'quit' or 'exit' to end the session.")
        print("="*60)
        
        # Interactive chat loop
        while True:
            # Get user input from command line
            user_input = input("\nYou: ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\nGoodbye! JIRA Assistant session ended.")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            print("\nJIRA Assistant is processing your request...")
            
            try:
                # For the first message in a thread, include system prompt
                # For subsequent messages, rely on memory
                messages = [SystemMessage(content=system_prompt)] if not memory else []
                messages.append({"role": "user", "content": user_input})
                
                response = await agent.ainvoke(
                    {"messages": messages},
                    config=config
                )

                # Extract assistant's response
                assistant_response = response["messages"][-1].content

                # Print the response in a formatted way
                print("\n" + "="*60)
                print("JIRA ASSISTANT:")
                print("="*60)
                print(assistant_response)
                print("="*60)
                
            except Exception as agent_error:
                print(f"\nError processing your request: {agent_error}")
                print("Please try rephrasing your request or check your JIRA configuration.")
                continue
        
        return True
        
    except Exception as e:
        # Handle any exceptions that occur during execution
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Ensure the client is properly closed even if an error occurs
        try:
            print("\nClosing MCP client...")
            # MultiServerMCPClient doesn't have a close method, it will clean up automatically
        except Exception as close_error:
            print(f"Error closing client: {close_error}")


async def main():
    """
    Main function to run the interactive JIRA assistant.
    """
    print("Starting JIRA MCP Agent Interactive Chat\n")
    
    result = await run_jira_assistant()
    
    print(f"\nFinal Result: {'SUCCESS - Working' if result else 'FAILED - Not working'}")


# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())