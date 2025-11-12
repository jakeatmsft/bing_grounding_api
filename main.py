from typing import Optional
from fastapi import FastAPI, Query
import os
import time

from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import BingGroundingTool
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Create the FastAPI application with optional metadata
app = FastAPI(
    title="My Search API",
    description="An example FastAPI application with a /search endpoint, complete with automatic Swagger docs at /docs.",
    version="1.0.0"
)


@app.get("/search", summary="Search Endpoint", description="Accepts a query string and returns search results.")
async def search(query: str = Query(..., description="Search query")):

    """
    Search endpoint that accepts a query string and returns search results.
    
    Args:
        query (str): The search query provided by the user.
        
    Returns:
        dict: A dictionary containing the search results.
    """ 
    print("Starting the Bing Grounding AI agent setup process.")  
  
    # Step 0: Validate environment variables  
    print("Step 0: Validating environment variables...")  
    project_conn_str = os.environ.get("PROJECT_CONNECTION_STRING")
    bing_connection_name = os.environ.get("BING_RESOURCE_NAME")
    agent_name = os.environ.get("AGENT_NAME")
    agent_instructions = os.environ.get("AGENT_INSTRUCTIONS")
    agent_llm = os.environ.get("MODEL_DEPLOYMENT_NAME", 'gpt-4.1')

    missing_vars = []
    if not project_conn_str:
        missing_vars.append("PROJECT_CONNECTION_STRING")
    if not bing_connection_name:
        missing_vars.append("BING_RESOURCE_NAME")
    if missing_vars:
        raise EnvironmentError(
            f"Missing environment variable(s): {', '.join(missing_vars)}"
        )
    print("Environment variables validated successfully.")

    try:
        # Step 1: Initialize the AI Project Client with default credentials  
        print("Step 1: Initializing Azure AI Project Client...")  
        credential = DefaultAzureCredential()  
        project_client = AIProjectClient(  
            credential=credential,  
            endpoint=project_conn_str  
        )  
        print("Azure AI Project Client initialized.")  
  
        with project_client:
            print("Step 2: Enabling Bing Grounding Tool...")
            bing_connection = project_client.connections.get(bing_connection_name)
            bing_tool = BingGroundingTool(connection_id=bing_connection.id)

            # # Stronger instructions
            # enforced_instructions = (
            #     "You are a factual assistant. "
            #     "For any question:\n"
            #     "1. Use the Bing grounding tool to search the web.\n"
            #     "2. Cite at least 1–3 source URLs at the end under 'Sources:'.\n"
            #     "If you cannot find data, clearly say so.\n"
            #     "Answer directly; do not start with greetings."
            # )
            # # Optionally override environment instructions:
            # final_instructions = enforced_instructions

            # Look for existing agent only if its instructions match our pattern; else recreate
            agents_list = list(project_client.agents.list_agents())
            agent = next((a for a in agents_list if a.name == agent_name), None)
            # if agent:
            #     # If existing agent has old, generic instructions, recreate
            #     if getattr(agent, "instructions", "")[:25] not in enforced_instructions[:25]:
            #         print("Existing agent instructions differ; creating a fresh agent.")
            #         agent = None

            if agent is None:
                agent = project_client.agents.create_agent(
                    model=agent_llm,
                    name=agent_name,
                    instructions=agent_instructions,
                    tools=bing_tool.definitions,
                    headers={"x-ms-enable-preview": "true"},
                    temperature=0,  # reduce small talk
                )
            print(f"Using agent ID: {agent.id}")

            # Step 4: Create thread
            thread = project_client.agents.threads.create()
            print(f"Thread ID: {thread.id}")

            # Step 5: Add user message - prepend directive to emphasize action
            print("Step 5: Adding user message to the thread...")
            user_message = project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            print(f"User message ID: {user_message.id}")

            # Step 6: Run agent (simple wait)
            run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
            print(f"Initial run status: {run.status}")

            wait_seconds = 7  # Slightly longer to allow tool call
            print(f"Waiting {wait_seconds}s for agent + Bing tool invocation...")
            time.sleep(wait_seconds)

            # Optional refresh
            try:
                run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)
                print(f"Run status after wait: {run.status}")
            except Exception as e:
                print(f"Run refresh failed: {e}")

            if run.status == "failed":
                return {
                    "query": query,
                    "status": run.status,
                    "error": str(getattr(run, "last_error", "Unknown error"))
                }

            # Step 7: Collect messages
            messages_list = list(project_client.agents.messages.list(thread_id=thread.id))

            # # Debug: Extract any tool call blocks
            # tool_calls_debug = []
            # for m in messages_list:
            #     if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
            #         for item in m.content:
            #             # Different SDK versions may label tool calls differently
            #             if hasattr(item, "tool_call") or getattr(item, "type", "") == "toolInvocation":
            #                 call_obj = getattr(item, "tool_call", None) or item
            #                 tool_calls_debug.append({
            #                     "tool_name": getattr(call_obj, "name", None),
            #                     "status": getattr(call_obj, "status", None),
            #                     "id": getattr(call_obj, "id", None),
            #                 })

            last_msg = next((m for m in reversed(messages_list) if m.role == "assistant"), None)

            assistant_text = ""
            if last_msg and last_msg.content:
                for item in last_msg.content:
                    if 'text' in item and 'value' in item['text'] and item['text']['value']:
                        assistant_text += item['text']['value'] + "\n"
            # citations = []
            # if last_msg and last_msg['content']:
            #     for item in last_msg['content']:
            #         if 'text' in item and 'value' in item['text'] and item['text']['value']:
            #             assistant_text += item['text']['value'] + "\n"

            #         # Collect citation annotations
            #         if 'annotations' in item['text']:
            #             for ann in item['text']['annotations']:
            #                 if 'url_citation' in ann:
            #                     citations.append({
            #                         "title": ann['url_citation']['title'],
            #                         "url": ann['url_citation']['url']
            #         })

            assistant_text = assistant_text.strip()
            if not assistant_text:
                print("Assistant produced no factual content; may need longer wait or instructions tweak.")

            return {
                "query": query,
                "agent_id": agent.id,
                "thread_id": thread.id,
                "run_id": getattr(run, "id", None),
                "run_status": getattr(run, "status", None),
                "assistant_response": assistant_text or None
            }
    except Exception as e:  
        print(f"An error occurred: {e}")
        return {"error": str(e)}