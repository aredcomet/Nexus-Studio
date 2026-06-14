import sys
import os
import requests
import json
import urllib.parse
import threading
import time
import mlx.core as mx
from transformers import AutoTokenizer

# Ensure the parent recurrentgemma directory is in the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DecoderCache
from generate import load_model
from mlx_lm.sample_utils import (
    apply_top_k,
    apply_top_p,
    apply_min_p,
    make_repetition_penalty,
)

# Configuration
CONFIG_PATH = "models/recurrentgemma-2b-it/config.json"
WEIGHTS_PATH = "weights/recurrentgemma-2b-it-mxfp4/weights.safetensors"
TOKENIZER_PATH = "models/recurrentgemma-2b-it"
SSE_URL = "http://127.0.0.1:8000/sse"

# Check fallbacks for weights
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = "weights/recurrentgemma-2b-it-4bit/weights.safetensors"
    if not os.path.exists(WEIGHTS_PATH):
        WEIGHTS_PATH = "weights/recurrentgemma-2b-it/weights.safetensors"

# Load the model
print("Loading RecurrentGemma model...")
model, config = load_model(CONFIG_PATH, WEIGHTS_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Global variables for MCP client
post_url = None
mcp_response = {}
mcp_response_event = {}

def read_sse(r):
    global post_url
    event = None
    for line in r.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('event:'):
                event = decoded.split('event:')[1].strip()
            elif decoded.startswith('data:'):
                data = decoded.split('data:')[1].strip()
                if event == 'endpoint' and not post_url:
                    post_url = urllib.parse.urljoin(SSE_URL, data)
                elif event == 'message':
                    msg = json.loads(data)
                    msg_id = msg.get('id')
                    if msg_id is not None:
                        mcp_response[msg_id] = msg
                        if msg_id in mcp_response_event:
                            mcp_response_event[msg_id].set()

# Connect to MCP server
print("Connecting to MCP Server at http://127.0.0.1:8000/sse...")
sse_res = requests.get(SSE_URL, stream=True)
sse_thread = threading.Thread(target=read_sse, args=(sse_res,), daemon=True)
sse_thread.start()

# Wait for post_url
for _ in range(50):
    if post_url:
        break
    time.sleep(0.1)

if not post_url:
    print("Error: Could not retrieve MCP POST endpoint. Exiting.")
    sys.exit(1)

print("MCP endpoint found:", post_url)

# Perform MCP Handshake
# 1. Initialize
init_id = 1
mcp_response_event[init_id] = threading.Event()
init_payload = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "recurrentgemma-client", "version": "1.0.0"}
    },
    "id": init_id
}
requests.post(post_url, json=init_payload)
mcp_response_event[init_id].wait(timeout=5)

# 2. Send initialized notification
initialized_payload = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
    "params": {}
}
requests.post(post_url, json=initialized_payload)
print("MCP Handshake completed successfully!")

# Local generate utility
def run_model_generation(prompt, max_tokens=150, temp=0.0):
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(formatted_prompt))[None, :]
    
    cache = DecoderCache(
        num_layers=model.config.num_hidden_layers,
        lru_width=model.config.lru_width,
        hidden_size=model.config.hidden_size
    )
    
    # Simple greedy decoding
    logits = model(input_ids, offset=0, cache=cache)
    token_logits = logits[:, -1, :]
    token = mx.argmax(token_logits[0], axis=-1).item()
    
    generated_tokens = [token]
    offset = input_ids.shape[1]
    
    for i in range(1, max_tokens):
        curr_input = mx.array([[token]])
        position_ids = mx.array([[offset]])
        logits = model(curr_input, position_ids=position_ids, offset=offset, cache=cache)
        token_logits = logits[:, -1, :]
        token = mx.argmax(token_logits[0], axis=-1).item()
        generated_tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
        offset += 1
        
    return tokenizer.decode(generated_tokens).strip()

# Construct the prompt with instructions and tools
mcp_tools_desc = """You have access to the following tools:
- get_current_date: Get the current local date and time of the host machine.
- search_documents: Search local documents. Parameters: {"query": "search query"}

To call a tool, you MUST reply in the exact format:
Thought: <reasoning about what tool to use>
Action: <name of the tool>
Action Input: <JSON-formatted arguments for the tool>

If you have the final answer, reply in the exact format:
Thought: <reasoning that you have the answer>
Final Answer: <your response to the user>"""

user_query = "What is today's date?"

react_prompt = f"{mcp_tools_desc}\n\nUser Query: {user_query}\nBegin!"

print(f"\nPrompting model with: '{user_query}'")
first_response = run_model_generation(react_prompt)
print("-" * 50)
print("Model Response:")
print(first_response)
print("-" * 50)

# Check if model chose to perform an action
if "Action:" in first_response:
    # Parse Action and Action Input
    action = None
    action_input = {}
    lines = first_response.split('\n')
    for idx, line in enumerate(lines):
        if line.startswith("Action:"):
            action = line.split("Action:")[1].strip()
            # Find the next Action Input line
            for j in range(idx + 1, len(lines)):
                if lines[j].startswith("Action Input:"):
                    raw_input = lines[j].split("Action Input:")[1].strip()
                    if "<eos>" in raw_input:
                        raw_input = raw_input.replace("<eos>", "").strip()
                    try:
                        action_input = json.loads(raw_input)
                    except Exception as e:
                        print("Failed to parse Action Input JSON:", e)
                    break
            break
                
    if action == "get_current_date":
        action_input = {}
        print(f"\n[Agent] Model triggered tool '{action}' with args {action_input}")
        
        # Execute tool call on MCP server
        call_id = 3
        mcp_response_event[call_id] = threading.Event()
        call_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_current_date",
                "arguments": action_input
            },
            "id": call_id
        }
        print("[Client] Calling tool on MCP server...")
        requests.post(post_url, json=call_payload)
        mcp_response_event[call_id].wait(timeout=5)
        
        result_msg = mcp_response.get(call_id, {})
        tool_result = "Unknown error"
        if "result" in result_msg and "content" in result_msg["result"]:
            content_list = result_msg["result"]["content"]
            if content_list and "text" in content_list[0]:
                tool_result = content_list[0]["text"]
                
        print("[Agent] Tool Output (Observation):", tool_result)
        
        # Feed observation back to the model
        final_prompt = f"{react_prompt}\n{first_response}\nObservation: {tool_result}\nBegin!"
        print("\nPrompting model with Observation...")
        final_response = run_model_generation(final_prompt)
        print("-" * 50)
        print("Final Model Response:")
        print(final_response)
        print("-" * 50)
else:
    print("\nModel did not choose to run any tool.")
