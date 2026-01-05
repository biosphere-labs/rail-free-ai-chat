# Rail Free AI Chat

A local AI chat application with text-to-speech and web search capabilities.

## Features

- **Multiple AI Models**: Switch between DeepSeek, Llama, Qwen, and Mixtral models
- **Voice Output**: Responses are read aloud using natural-sounding voices
- **Web Search**: Ask questions about current events using Brave Search
- **Conversation History**: Your chats are saved locally and can be resumed
- **Socratic Challenger**: Optionally fact-checks responses for accuracy

## Quick Start

1. Copy `.env.example` to `.env` and add your API keys
2. Install dependencies: `pip install -e .`
3. Run: `chainlit run app.py`
4. Open http://localhost:8000 in your browser

## Epic Executor

Also includes a CLI tool for executing task files in parallel. Point it at a folder of numbered markdown task files and it will:

- Build a dependency graph from task references
- Run independent tasks in parallel using AI agents
- Verify each task's acceptance criteria before marking complete

```bash
# Preview the execution plan
python -m epic_executor.cli /path/to/tasks --dry-run

# Execute tasks
python -m epic_executor.cli /path/to/tasks
```

## Requirements

- Python 3.10+
- [DeepInfra API key](https://deepinfra.com/) for AI models
- [Brave Search API key](https://brave.com/search/api/) (optional, for web search)

## License

MIT

---

*Built with [Claude Code](https://claude.ai/code) in 1 hour.*
