from openai import OpenAI
"""
FastAPI application for AI-powered travel advisor with LLM, RAG, and agentic frameworks.

This module provides a travel advisor API that supports multiple execution modes:
- LLM: Direct language model queries
- Agentic: Tool-based agent that validates cities before providing travel advice

The application integrates with:
- OpenAI-compatible LLM endpoints (default: Ollama)
- Dynatrace OpenTelemetry for distributed tracing and observability
- Traceloop for LLM-specific instrumentation

Environment Variables:
    AI_MODEL (str): Language model name (default: "orca-mini:3b")
    AI_EMBEDDING_MODEL (str): Embedding model name (default: "orca-mini:3b")
    OPENAI_BASE_URL (str): LLM endpoint URL (default: "http://localhost:11434")
    OPENAI_API_KEY (str): API key for LLM (default: "ollama")
    OTEL_ENDPOINT (str): Dynatrace OTLP API endpoint for telemetry
    DT_API_TOKEN (str): Dynatrace API token for authentication

The agentic executor uses a manual tool orchestration pattern:
1. Validates if input is a valid city name
2. If valid, generates travel advice
3. If invalid, provides a humorous excuse

All endpoints are instrumented with OpenTelemetry spans for distributed tracing.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from opentelemetry import trace
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task
from colorama import Fore

# disable traceloop telemetry
os.environ["TRACELOOP_TELEMETRY"] = "false"

def read_token():
    return os.environ.get("API_TOKEN", read_secret("token"))

def read_endpoint():
    return os.environ.get("OTEL_ENDPOINT", read_secret("endpoint"))

def read_secret(secret: str):
    try:
        with open(f"/etc/secrets/{secret}", "r") as f:
            return f.read().rstrip()
    except Exception as e:
        print(f"No {secret} was provided")
        print(e)
        return ""

# By default we use orca-mini:3b because it's small enough to run easily on codespace
# Make sure if you change this, you need to also change the deployment script
AI_MODEL = os.environ.get("AI_MODEL", "orca-mini:3b")
AI_EMBEDDING_MODEL = os.environ.get("AI_EMBEDDING_MODEL", "orca-mini:3b")

## Configuration of OpenAI-compatible endpoint & Weaviate
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434"),
)
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", "ollama"),
)
if not OPENAI_BASE_URL.endswith("/v1"):
    OPENAI_BASE_URL = f"{OPENAI_BASE_URL.rstrip('/')}/v1"

print(f"{Fore.GREEN} Connecting to OpenAI-compatible LLM ({AI_MODEL}): {OPENAI_BASE_URL} {Fore.RESET}")

openai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

llm = ChatOpenAI(
    model=AI_MODEL,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

MAX_PROMPT_LENGTH = 50

# Initialise the logger
logging.basicConfig(level=logging.INFO, filename="run.log")
logger = logging.getLogger(__name__)

#################
# CONFIGURE TRACELOOP & OTel

# Prefer direct Dynatrace OTLP endpoint when present, otherwise fallback to OTEL_ENDPOINT/secret.
OTEL_ENDPOINT = os.environ.get("DT_OTLP_API_ENDPOINT") 
#OTEL_ENDPOINT = os.environ.get("OTEL_ENDPOINT")
print(f"OTEL_ENDPOINT = {OTEL_ENDPOINT}")

#TOKEN = read_token()
TOKEN=os.environ.get("DT_API_TOKEN")
headers = {"Authorization": f"Api-Token {TOKEN}"}

otel_tracer = trace.get_tracer("travel-advisor")

# Initialize OpenLLMetry
Traceloop.init(
    app_name="ai-travel-advisor",
    api_endpoint=OTEL_ENDPOINT,
    disable_batch=True, # This is recomended for testing but NOT for production
    headers=headers,
)

def openai_generate(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=AI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""

##########
# Agentic Tools

import re
regex = re.compile('[^a-zA-Z]')

@tool
@task(name="tool_valid_city")
def valid_city(city: str) -> bool:
    """Returns if the input is a valid city"""
    prompt = f"Is {city} a city? respond ONLY with yes or no."
    response = openai_generate(prompt)
    response = regex.sub('', response).lower()
    return response == "yes" or response.startswith("yes")

@tool(return_direct=True)
@task(name="tool_travel_advice")
def travel_advice(city: str) -> str:
    """Provide travel advice for the given city"""
    prompt = f"Give travel advise in a paragraph of max 50 words about {city}"
    response = openai_generate(prompt)
    return response

@tool(return_direct=True)
@task(name="tool_excuse")
def excuse(city: str) -> str:
    """Returns an excuse why it cannot provide an answer"""
    prompt = f"Provide an excuse on why you cannot provide a travel advice about {city}"
    return openai_generate(prompt)


############
# Setup the endpoints and LangChain

def prep_agent_executor() -> AgentExecutor:
    tools = [valid_city, travel_advice, excuse]
    system_prompt = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

MANDATORY WORKFLOW RULES:
1) You MUST call "valid_city" first, exactly once.
2) If valid_city result is true/yes, your next and only tool call MUST be "travel_advice".
3) If valid_city result is false/no, your next and only tool call MUST be "excuse".
4) Never call "travel_advice" or "excuse" before calling "valid_city".
5) Never skip tools and never answer directly before completing this workflow.
6) Never call the same tool more than once.
7) After calling "travel_advice" or "excuse", immediately finish; do not call any additional tools.

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation.
Do not add any text outside a single JSON action blob.'''

    human_prompt = '''
{input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)'''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human_prompt),
        ]
    )
    agent = create_structured_chat_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        max_execution_time=20,
        early_stopping_method="force",
    )

app = FastAPI()
agentic_executor = prep_agent_executor()

####################################
@app.get("/api/v1/completion")
def submit_completion(framework: str, prompt: str):
    with otel_tracer.start_as_current_span(name="/api/v1/completion", kind=trace.SpanKind.SERVER) as span:
        if framework == "llm":
            return llm_chat(prompt)
        if framework == "agentic":
            return agentic_chat(prompt)
        return {"message": "invalid Mode"}


@task(name="llm_chat")
def llm_chat(prompt: str):
    prompt = f"Give travel advise in a paragraph of max 50 words about {prompt}"
    return {"message": openai_generate(prompt)}


@task(name="agentic_chat")
def agentic_chat(prompt: str):
    city = prompt.strip()
    is_valid = valid_city.invoke(city)
    if not is_valid:
        return {"message": excuse.invoke(city)}

    task_input = f"{city} is a valid city. Use travel_advice exactly once and return the final answer."
    response = agentic_executor.invoke({
        "input": task_input,
    })
    output = response.get("output", "")
    if "Agent stopped due to iteration limit" in output or "time limit" in output:
        logger.warning(f"Agent fallback triggered for prompt: {prompt}")
        fallback_prompt = f"Give travel advise in a paragraph of max 50 words about {prompt}"
        return {"message": openai_generate(fallback_prompt)}
    return {"message": output}

####################################
@app.get("/api/v1/thumbsUp")
@otel_tracer.start_as_current_span("/api/v1/thumbsUp")
def thumbs_up(prompt: str):
    logger.info(f"Positive user feedback for search term: {prompt}")


@app.get("/api/v1/thumbsDown")
@otel_tracer.start_as_current_span("/api/v1/thumbsDown")
def thumbs_down(prompt: str):
    logger.info(f"Negative user feedback for search term: {prompt}")


if __name__ == "__main__":

    # Mount static files at the root
    app.mount("/", StaticFiles(directory="./public", html=True), name="public")

    # Run the app using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)