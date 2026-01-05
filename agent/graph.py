import os
import httpx
from typing import AsyncGenerator, TypedDict, Annotated, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .tools import get_tools

load_dotenv()


async def brave_search(query: str, count: int = 3) -> list[dict]:
    """Search Brave and return results.

    Args:
        query: Search query
        count: Number of results (max 3 to stay fast)

    Returns:
        List of {title, url, description} dicts
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return []

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key},
                params={"q": query, "count": count},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:count]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                })
            return results
        except Exception as e:
            print(f"Brave search error: {e}")
            return []

CHALLENGER_PROMPT = """You are a Socratic challenger with access to current web search results. Review the assistant's response and:

1. **Fact-check**: Use the search context to verify claims. If something contradicts current information or is outdated, point it out with sources.

2. **Simplicity check**: Is there a clearer, more direct way to explain this? Cut jargon, unnecessary complexity, or tangents.

3. **80-20 focus**: What's the essential insight? If 20% of the answer delivers 80% of the value, highlight what matters most and what can be trimmed.

4. **Question assumptions**: What's being taken for granted? Ask "What if..." or "Have you considered the opposite?"

5. **Stay on topic**: Does the response answer what was actually asked, or did it drift? Redirect if needed.

6. **Be specific**: If the response is vague or generic, ask for concrete examples, evidence, or trade-offs.

If the response is clear, accurate, focused, and directly answers the question - approve it.

Respond in one of two formats:

APPROVED: [Brief reason why the response is good]

Or:

CHALLENGE: [Your specific question or pushback. Reference search results if relevant. Be actionable - say exactly what to reconsider, clarify, or simplify.]

Be concise. One challenge at a time."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    challenge_count: int
    challenger_enabled: bool
    search_context: str  # Web search results for fact-checking
    current_stage: str  # For progress reporting
    challenge_feedback: str  # Feedback from challenger to refine response


def get_llm() -> ChatOpenAI:
    """Create the LLM instance configured for DeepInfra."""
    return ChatOpenAI(
        model=os.environ.get("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V3"),
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
        temperature=0.7,
        max_tokens=4096,
    )


MAX_REFINEMENT_LOOPS = 2

def create_agent(tools: list = None, checkpointer=None, challenger_enabled: bool = True):
    """Create the LangGraph agent with Socratic challenger and refinement loop.

    Args:
        tools: List of tools for the agent. Defaults to built-in tools.
        checkpointer: State checkpointer for conversation persistence.
        challenger_enabled: Whether to enable the Socratic challenger step.

    Returns:
        The compiled agent graph.
    """
    llm = get_llm()

    if tools is None:
        tools = get_tools()

    if checkpointer is None:
        checkpointer = MemorySaver()

    # Create the base ReAct agent
    react_agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    # Build the wrapper graph with challenger
    builder = StateGraph(AgentState)

    async def call_agent(state: AgentState) -> dict:
        """Call the ReAct agent, incorporating any challenge feedback."""
        messages = list(state["messages"])

        # If there's challenge feedback, add it as context for refinement
        challenge_feedback = state.get("challenge_feedback", "")
        if challenge_feedback:
            # Add the challenge as a system instruction for refinement
            refinement_msg = HumanMessage(
                content=f"Please revise your previous response based on this feedback:\n\n{challenge_feedback}"
            )
            messages.append(refinement_msg)

        result = await react_agent.ainvoke({"messages": messages})
        return {
            "messages": result["messages"],
            "current_stage": "drafting",
            "challenge_feedback": "",  # Clear feedback after use
        }

    async def gather_search_context(state: AgentState) -> dict:
        """Search for context to fact-check the response."""
        if not state.get("challenger_enabled", True):
            return {"search_context": "", "current_stage": "complete"}

        # Get the last AI message
        last_ai_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai_msg = msg
                break

        if not last_ai_msg:
            return {"search_context": "", "current_stage": "complete"}

        # Get the user's question
        user_question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        # Generate a search query from the response
        query_prompt = f"Extract a concise search query (3-6 words) to fact-check this response. Just output the query, nothing else.\n\nUser question: {user_question}\n\nResponse to fact-check: {last_ai_msg.content[:500]}"

        query_response = await llm.ainvoke([HumanMessage(content=query_prompt)])
        search_query = query_response.content.strip().strip('"')

        # Search Brave
        results = await brave_search(search_query, count=3)

        if not results:
            return {"search_context": "No search results available.", "current_stage": "fact-checking"}

        # Format results
        context_parts = [f"Web search for '{search_query}':"]
        for r in results:
            context_parts.append(f"- {r['title']}: {r['description']}")

        return {"search_context": "\n".join(context_parts), "current_stage": "fact-checking"}

    async def challenge_response(state: AgentState) -> dict:
        """Socratic challenger reviews the response with search context."""
        if not state.get("challenger_enabled", True):
            return {"current_stage": "complete", "challenge_feedback": ""}

        # Get the last AI message
        last_ai_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai_msg = msg
                break

        if not last_ai_msg:
            return {"current_stage": "complete", "challenge_feedback": ""}

        # Get conversation context (last few messages)
        recent_messages = state["messages"][-6:]
        search_context = state.get("search_context", "")

        challenge_content = f"Original conversation:\n{_format_messages(recent_messages)}\n\nAssistant's response to review:\n{last_ai_msg.content}"

        if search_context:
            challenge_content += f"\n\n---\n{search_context}"

        challenge_messages = [
            SystemMessage(content=CHALLENGER_PROMPT),
            HumanMessage(content=challenge_content),
        ]

        response = await llm.ainvoke(challenge_messages)
        challenge_text = response.content.strip()

        if challenge_text.startswith("APPROVED"):
            return {"current_stage": "complete", "challenge_feedback": ""}

        if challenge_text.startswith("CHALLENGE:"):
            challenge_question = challenge_text.replace("CHALLENGE:", "").strip()
            return {
                "challenge_feedback": challenge_question,
                "challenge_count": state.get("challenge_count", 0) + 1,
                "current_stage": "refining",
            }

        return {"current_stage": "complete", "challenge_feedback": ""}

    def should_challenge(state: AgentState) -> Literal["challenge", "end"]:
        """Decide whether to run the challenger."""
        if not state.get("challenger_enabled", True):
            return "end"
        # Max refinement loops reached
        if state.get("challenge_count", 0) >= MAX_REFINEMENT_LOOPS:
            return "end"
        # Check if last message is from AI
        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            return "challenge"
        return "end"

    def should_refine(state: AgentState) -> Literal["refine", "end"]:
        """Decide whether to loop back for refinement."""
        challenge_feedback = state.get("challenge_feedback", "")
        challenge_count = state.get("challenge_count", 0)

        # If we have feedback and haven't hit max loops, refine
        if challenge_feedback and challenge_count < MAX_REFINEMENT_LOOPS:
            return "refine"
        return "end"

    builder.add_node("agent", call_agent)
    builder.add_node("search", gather_search_context)
    builder.add_node("challenger", challenge_response)

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_challenge,
        {"challenge": "search", "end": END},
    )
    builder.add_edge("search", "challenger")
    builder.add_conditional_edges(
        "challenger",
        should_refine,
        {"refine": "agent", "end": END},
    )

    graph = builder.compile(checkpointer=checkpointer)
    return graph


def _format_messages(messages: list) -> str:
    """Format messages for the challenger context."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content[:500]}...")
    return "\n".join(formatted)


async def run_agent(
    agent,
    message: str,
    thread_id: str = "default",
    challenger_enabled: bool = True,
) -> AsyncGenerator[str, None]:
    """Run the agent and stream the response.

    Args:
        agent: The compiled agent graph.
        message: The user's message.
        thread_id: Conversation thread ID for state persistence.
        challenger_enabled: Whether to enable the Socratic challenger.

    Yields:
        Response chunks as they're generated.
    """
    config = {"configurable": {"thread_id": thread_id}}

    async for event in agent.astream_events(
        {
            "messages": [HumanMessage(content=message)],
            "challenge_count": 0,
            "challenger_enabled": challenger_enabled,
            "search_context": "",
        },
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content

        elif kind == "on_tool_end":
            tool_output = event["data"].get("output", "")
            if tool_output:
                yield f"\n[Tool output: {tool_output[:200]}{'...' if len(str(tool_output)) > 200 else ''}]\n"


async def run_agent_simple(
    agent,
    message: str,
    thread_id: str = "default",
    challenger_enabled: bool = True,
) -> str:
    """Run the agent and return the complete response.

    Args:
        agent: The compiled agent graph.
        message: The user's message.
        thread_id: Conversation thread ID for state persistence.
        challenger_enabled: Whether to enable the Socratic challenger.

    Returns:
        The complete response string.
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "challenge_count": 0,
            "challenger_enabled": challenger_enabled,
            "search_context": "",
            "current_stage": "drafting",
            "challenge_feedback": "",
        },
        config=config,
    )

    # Return only the final AI message (the refined response)
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return ""


async def run_agent_with_progress(
    agent,
    message: str,
    thread_id: str = "default",
    challenger_enabled: bool = True,
    on_stage_change: callable = None,
) -> str:
    """Run the agent with progress callbacks.

    Args:
        agent: The compiled agent graph.
        message: The user's message.
        thread_id: Conversation thread ID for state persistence.
        challenger_enabled: Whether to enable the Socratic challenger.
        on_stage_change: Async callback called with (stage_name, stage_description).

    Returns:
        The complete response string.
    """
    config = {"configurable": {"thread_id": thread_id}}

    stage_descriptions = {
        "drafting": "Drafting response...",
        "fact-checking": "Fact-checking with web search...",
        "challenging": "Reviewing for clarity and accuracy...",
        "refining": "Refining based on feedback...",
        "complete": "Complete",
    }

    last_stage = None

    async for event in agent.astream(
        {
            "messages": [HumanMessage(content=message)],
            "challenge_count": 0,
            "challenger_enabled": challenger_enabled,
            "search_context": "",
            "current_stage": "drafting",
            "challenge_feedback": "",
        },
        config=config,
        stream_mode="values",
    ):
        current_stage = event.get("current_stage", "")
        if current_stage and current_stage != last_stage and on_stage_change:
            description = stage_descriptions.get(current_stage, current_stage)
            await on_stage_change(current_stage, description)
            last_stage = current_stage

    # Get final state
    final_state = await agent.aget_state(config)
    messages = final_state.values.get("messages", [])

    # Return only the final AI message
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return ""
