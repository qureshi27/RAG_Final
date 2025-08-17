import os
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
import json
from openai import OpenAI

load_dotenv()

client = OpenAI()
EMBEDDINGS_DIR = "./uploaded_files/embeddings"

def load_query_engine_for_user(email):
    # Now ignore email!
    if not os.path.exists(EMBEDDINGS_DIR):
        raise FileNotFoundError(f"No global embeddings found.")
    storage_context = StorageContext.from_defaults(persist_dir=EMBEDDINGS_DIR)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(response_mode="compact", similarity_top_k=3)
    return query_engine





import pandas as pd

def store_chat_history(chat_messages, session_id, csv_file="chathistory_service.csv"):
    session_id = str(session_id)
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=["id", "chat_history"]).to_csv(csv_file, index=False)
    chat_df = pd.read_csv(csv_file, dtype={"id": str})
    if session_id in chat_df["id"].values:
        chat_df.loc[chat_df["id"] == session_id, "chat_history"] = str(chat_messages)
    else:
        new_row = {"id": session_id, "chat_history": str(chat_messages)}
        chat_df = pd.concat([chat_df, pd.DataFrame([new_row])], ignore_index=True)
    chat_df.to_csv(csv_file, index=False)

def get_chat_by_id(session_id, csv_file="chathistory_service.csv"):
    try:
        session_id = str(session_id)
        history = pd.read_csv(csv_file, dtype={"id": str})
        if session_id in history["id"].values:
            chat_history_str = history.loc[history["id"] == session_id, "chat_history"].values[0]
            chat_history = eval(chat_history_str)
            return chat_history
        else:
            return []
    except Exception:
        return []


def retrieve_with_prompt(email: str, prompt: str, similarity_threshold: float = 0.5):
    engine = load_query_engine_for_user(email)
    response = engine.query(prompt)
    response_text = response.response

    filtered_nodes = [
        node for node in response.source_nodes
        if node.score is not None and node.score >= similarity_threshold
    ]
    sources = [node.node.metadata.get("source", "Unknown") for node in filtered_nodes] or ["No sources above threshold"]

    return {
        "response": response_text,
        "sources": sources
    }




system_message = {
    "role": "system",
    "content": (
        "You are an AI assistant. Handle greetings and general queries naturally. "
        "For anything factual or document-based, use the 'retrieve_with_prompt' tool. "
        "Never mention tools or the knowledge base in your response."
    )
}
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_with_prompt",
            "description": "Fetch an answer from the knowledge base for the user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

async def ai_assistant(query, email, session_id):
    messages = []
    CSV_FILE = "chathistory_service.csv"

    if os.path.exists(CSV_FILE):
        chat_history = get_chat_by_id(session_id, csv_file=CSV_FILE)
        for msg in chat_history:
            messages.append(msg)

    if not messages:
        messages.append(system_message)

    messages.append({"role": "user", "content": str(query)})

    # Call OpenAI with tools enabled
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls
    text_response = message.content

    # Direct answer from GPT-4o (no tool call)
    if text_response:
        messages.append({"role": "assistant", "content": text_response})
        store_chat_history(messages, session_id, csv_file=CSV_FILE)
        return {"response": text_response}

    # If tool was called
    if tool_calls:
        tool_name = tool_calls[0].function.name
        tool_args = json.loads(tool_calls[0].function.arguments)
        if tool_name == "retrieve_with_prompt":
            tool_result = retrieve_with_prompt(email, tool_args["query"])
            messages.append({"role": "assistant", "content": tool_result.get("response", "")})
            store_chat_history(messages, session_id, csv_file=CSV_FILE)
            return {
                "response": tool_result.get("response", ""),
                "source": tool_result.get("sources", [])
            }
