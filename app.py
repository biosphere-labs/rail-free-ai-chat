import os
import sqlite3
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv

load_dotenv()

# Initialize SQLite schema for Chainlit persistence
DB_PATH = "chat_history.db"

def init_db():
    """Create required tables for Chainlit SQLAlchemy data layer."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            "id" TEXT PRIMARY KEY,
            "identifier" TEXT NOT NULL UNIQUE,
            "metadata" TEXT NOT NULL DEFAULT '{}',
            "createdAt" TEXT
        );
        CREATE TABLE IF NOT EXISTS threads (
            "id" TEXT PRIMARY KEY,
            "createdAt" TEXT,
            "name" TEXT,
            "userId" TEXT,
            "userIdentifier" TEXT,
            "tags" TEXT,
            "metadata" TEXT,
            FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS steps (
            "id" TEXT PRIMARY KEY,
            "name" TEXT NOT NULL,
            "type" TEXT NOT NULL,
            "threadId" TEXT NOT NULL,
            "parentId" TEXT,
            "streaming" INTEGER NOT NULL,
            "waitForAnswer" INTEGER,
            "isError" INTEGER,
            "metadata" TEXT,
            "tags" TEXT,
            "input" TEXT,
            "output" TEXT,
            "createdAt" TEXT,
            "start" TEXT,
            "end" TEXT,
            "generation" TEXT,
            "showInput" TEXT,
            "language" TEXT,
            "indent" INTEGER,
            FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS elements (
            "id" TEXT PRIMARY KEY,
            "threadId" TEXT,
            "type" TEXT,
            "url" TEXT,
            "chainlitKey" TEXT,
            "name" TEXT,
            "display" TEXT,
            "objectKey" TEXT,
            "size" TEXT,
            "page" INTEGER,
            "language" TEXT,
            "forId" TEXT,
            "mime" TEXT,
            "props" TEXT
        );
        CREATE TABLE IF NOT EXISTS feedbacks (
            "id" TEXT PRIMARY KEY,
            "forId" TEXT NOT NULL,
            "threadId" TEXT NOT NULL,
            "value" INTEGER NOT NULL,
            "comment" TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

# SQLite persistence for chat history
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")


@cl.header_auth_callback
def header_auth_callback(headers: dict) -> cl.User | None:
    # No auth required - return default user for local use
    return cl.User(identifier="local_user")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Restore agent state when resuming a thread
    agent = create_agent(tools=get_tools())
    cl.user_session.set("agent", agent)
    cl.user_session.set("thread_id", thread["id"])


from agent import create_agent, run_agent_with_progress
from agent.tools import get_tools
from agent.mcp_config import get_enabled_servers
from tts import speak_to_file, POPULAR_VOICES, DEFAULT_VOICE

# Available models on DeepInfra
AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

DEFAULT_MODEL = os.environ.get("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V3")


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Get initial values from environment
    initial_model = DEFAULT_MODEL
    initial_api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    initial_brave_key = os.environ.get("BRAVE_API_KEY", "")

    # Store settings in session
    cl.user_session.set("model", initial_model)
    cl.user_session.set("api_key", initial_api_key)
    cl.user_session.set("brave_api_key", initial_brave_key)
    cl.user_session.set("thread_id", cl.context.session.id)
    cl.user_session.set("tts_enabled", True)
    cl.user_session.set("voice", DEFAULT_VOICE)
    cl.user_session.set("challenger_enabled", True)

    # Create agent for this session
    agent = create_agent(tools=get_tools())
    cl.user_session.set("agent", agent)

    # Show settings
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model",
                label="Model",
                values=AVAILABLE_MODELS,
                initial_value=initial_model if initial_model in AVAILABLE_MODELS else AVAILABLE_MODELS[0],
            ),
            cl.input_widget.TextInput(
                id="api_key",
                label="DeepInfra API Key",
                initial=initial_api_key[:8] + "..." if len(initial_api_key) > 8 else "",
                placeholder="Enter API key (leave empty to use .env)",
            ),
            cl.input_widget.TextInput(
                id="brave_api_key",
                label="Brave Search API Key",
                initial=initial_brave_key[:8] + "..." if len(initial_brave_key) > 8 else "",
                placeholder="Enter Brave API key (leave empty to use .env)",
            ),
            cl.input_widget.Switch(
                id="tts_enabled",
                label="Enable Text-to-Speech",
                initial=True,
            ),
            cl.input_widget.Select(
                id="voice",
                label="TTS Voice",
                values=[name for name, label in POPULAR_VOICES],
                initial_value=DEFAULT_VOICE,
            ),
            cl.input_widget.Switch(
                id="challenger_enabled",
                label="Enable Socratic Challenger",
                initial=True,
            ),
        ]
    ).send()



@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes."""
    cl.user_session.set("tts_enabled", settings.get("tts_enabled", True))
    cl.user_session.set("voice", settings.get("voice", DEFAULT_VOICE))
    cl.user_session.set("challenger_enabled", settings.get("challenger_enabled", True))

    # Check if model changed
    new_model = settings.get("model")
    old_model = cl.user_session.get("model")

    # Check if API key changed (ignore masked values)
    new_api_key = settings.get("api_key", "")
    if not new_api_key.endswith("..."):
        cl.user_session.set("api_key", new_api_key)
        os.environ["DEEPINFRA_API_KEY"] = new_api_key

    new_brave_key = settings.get("brave_api_key", "")
    if not new_brave_key.endswith("..."):
        cl.user_session.set("brave_api_key", new_brave_key)
        os.environ["BRAVE_API_KEY"] = new_brave_key

    # Recreate agent if model changed
    if new_model and new_model != old_model:
        cl.user_session.set("model", new_model)
        os.environ["DEEPINFRA_MODEL"] = new_model
        agent = create_agent(tools=get_tools())
        cl.user_session.set("agent", agent)
        await cl.Message(content=f"Switched to model: **{new_model}**").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    import uuid
    from pathlib import Path

    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")
    tts_enabled = cl.user_session.get("tts_enabled", True)
    voice = cl.user_session.get("voice", DEFAULT_VOICE)
    challenger_enabled = cl.user_session.get("challenger_enabled", True)

    # Create a status message that will show progress
    status_msg = cl.Message(content="⏳ Processing...")
    await status_msg.send()

    # Collect progress updates (detail, full_content)
    progress_items = []

    async def on_progress(stage: str, detail: str, full_content: str = ""):
        progress_items.append((detail, full_content))
        # Update the status message with accumulated progress
        status_msg.content = "⏳ " + detail
        await status_msg.update()

    # Get response from agent with progress
    response = await run_agent_with_progress(
        agent,
        message.content,
        thread_id=thread_id,
        challenger_enabled=challenger_enabled,
        on_progress=on_progress,
    )

    # Build the final message with thinking process collapsed at the top
    # Each thinking item is expandable to show full content
    final_content = ""
    if progress_items and challenger_enabled:
        final_content = "<details>\n<summary>💭 Thinking process</summary>\n\n"
        for detail, full_content in progress_items:
            if full_content and full_content != detail:
                # Make expandable item with full content
                final_content += f"<details>\n<summary>{detail}</summary>\n\n{full_content}\n\n</details>\n"
            else:
                # Simple item without expansion
                final_content += f"- {detail}\n"
        final_content += "\n</details>\n\n"
    final_content += response

    # Show response immediately (don't wait for TTS)
    status_msg.content = final_content
    await status_msg.update()

    # Generate TTS in background if enabled (user can click play when ready)
    if tts_enabled and response:
        tts_text = _strip_markdown(response)
        if tts_text:
            try:
                # Prepare audio file path
                audio_dir = Path(".files/audio")
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"{uuid.uuid4()}.mp3"

                # Stream TTS directly to file (chunks written as they arrive)
                await speak_to_file(tts_text, audio_path, voice)

                # Add audio element (no auto-play - user clicks to play)
                audio_element = cl.Audio(
                    name="🔊 Listen",
                    path=str(audio_path),
                    display="inline",
                    auto_play=False,
                )
                status_msg.elements = [audio_element]
                await status_msg.update()
            except Exception as e:
                print(f"TTS error: {e}")


def _strip_markdown(text: str) -> str:
    """Strip markdown and LaTeX formatting for cleaner TTS."""
    import re

    # Remove LaTeX display math ($$...$$, \[...\])
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    # Remove LaTeX inline math ($...$, \(...\))
    text = re.sub(r"\$[^$]+\$", "", text)
    text = re.sub(r"\\\([\s\S]*?\\\)", "", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove bold/italic
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    # Remove headers
    text = re.sub(r"#+\s*", "", text)
    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
