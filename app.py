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
            "mime" TEXT
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


from agent import create_agent, run_agent_simple
from agent.tools import get_tools
from agent.mcp_config import get_enabled_servers
from tts import speak, POPULAR_VOICES, DEFAULT_VOICE

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

    # Get response from agent
    response = await run_agent_simple(
        agent,
        message.content,
        thread_id=thread_id,
        challenger_enabled=challenger_enabled,
    )

    # Generate TTS if enabled
    audio_element = None
    if tts_enabled and response:
        # Strip markdown for cleaner TTS
        tts_text = _strip_markdown(response)
        if tts_text:
            try:
                audio_bytes = await speak(tts_text, voice)
                # Save audio to file for persistence
                audio_dir = Path(".files/audio")
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"{uuid.uuid4()}.mp3"
                audio_path.write_bytes(audio_bytes)

                # Create audio element with file path
                audio_element = cl.Audio(
                    name="response.mp3",
                    path=str(audio_path),
                    display="inline",
                    auto_play=True,
                )
            except Exception as e:
                print(f"TTS error: {e}")

    # Create and send complete message with audio
    msg = cl.Message(
        content=response,
        elements=[audio_element] if audio_element else None,
    )
    await msg.send()


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
