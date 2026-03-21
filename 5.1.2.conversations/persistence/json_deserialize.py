import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessagesTypeAdapter
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)


def main():
    history_path = Path(__file__).parent.parent.resolve() / "data" / "history.json"
    with open(history_path, "rb") as f:
        history = json.load(f)

    history = ModelMessagesTypeAdapter.validate_python(history)
    while True:
        user_message = input(">>")
        if user_message == "q":
            break
        result = agent.run_sync(user_message, message_history=history)
        print(result.output)
        history.extend(result.new_messages())


if __name__ == "__main__":
    main()

