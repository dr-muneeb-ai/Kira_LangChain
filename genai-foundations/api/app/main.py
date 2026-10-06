import os

from dotenv  import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

LLM = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    temperature=0.7,
)

history = []

while True:
    user_input = input(">>>> ")

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit"]:
        break

    history.append(HumanMessage(content=user_input))

    response = LLM.invoke(history)

    history.append(AIMessage(content=response.content))

    print("\n========== FULL HISTORY ==========")

    for message in history:
        if isinstance(message, HumanMessage):
            print(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")

    print("==================================\n")