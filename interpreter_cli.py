import os
from dotenv import load_dotenv
from interpreter import interpreter

load_dotenv()

# --- Configuration ---
interpreter.api_key = os.environ["DEEPINFRA_API_KEY"]
interpreter.model = os.environ.get(
    "DEEPINFRA_MODEL",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
)

# DeepInfra uses an OpenAI-compatible API
interpreter.api_base = "https://api.deepinfra.com/v1/openai"

# Safety / verbosity
interpreter.auto_run = False
interpreter.verbose = True
interpreter.context_window = 8192

def main():
    print("DeepInfra + Open Interpreter CLI")
    print(f"Model: {interpreter.model}")
    print("Type 'exit' or Ctrl-D to quit.\n")

    while True:
        try:
            user_input = input(">>> ")
        except EOFError:
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        interpreter.chat(user_input)

if __name__ == "__main__":
    main()
