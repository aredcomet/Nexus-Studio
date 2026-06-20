# ==============================================================================
# 1. Path Initialization (Must run before loading engine submodules)
# ==============================================================================
import os
import sys

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

engines_dir = os.path.join(workspace_root, "engines")
if engines_dir not in sys.path:
    sys.path.append(engines_dir)

# ==============================================================================
# 2. Standard Library Imports
# ==============================================================================
import argparse
import asyncio
import gc
import json
import re
import subprocess
import threading
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Union

# ==============================================================================
# 3. Third-Party Imports
# ==============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel
import requests
from transformers import AutoProcessor, AutoTokenizer
import uvicorn

# Optional import with fallback
try:
    import psutil
except ImportError:
    psutil = None

# ==============================================================================
# 4. mlx-vlm and mlx-lm Imports
# ==============================================================================
import mlx_vlm
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, make_repetition_penalty


app = FastAPI(title="MLX Multi-Model Gateway API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model, config, tokenizer/processor state
model = None
tokenizer = None
processor = None
draft_model = None
draft_kind = None
draft_block_size = None
model_type = None
model_name = "None"
loaded_paths = {}
mcp_clients = {}
pending_approvals = {}

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "default"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    repeat_penalty: float = 1.1
    repeat_context: int = 20
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = None

class ModelLoadRequest(BaseModel):
    model_type: str  # "recurrentgemma", "t5v2", or "standard"
    config_path: str
    weights_path: str
    tokenizer_path: str
    adapter_path: Optional[str] = None
    attention_window: Optional[int] = None
    chat_template_path: Optional[str] = None
    ignore_layers: Optional[List[str]] = None
    draft_model_path: Optional[str] = None
    draft_kind: Optional[str] = None
    draft_block_size: Optional[int] = None

def clean_messages_for_gemma(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges system messages into the first user message for compatibility with Gemma's template."""
    cleaned = []
    system_content = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            from mlx_vlm.prompt_utils import extract_text_from_content
            content_str = extract_text_from_content(content)
        else:
            content_str = str(content)
            
        if role == "system":
            system_content = content_str
        elif role == "user":
            if system_content:
                cleaned.append({"role": "user", "content": f"{system_content}\n\n{content_str}"})
                system_content = ""
            else:
                cleaned.append({"role": "user", "content": content_str})
        else:
            cleaned.append({"role": role, "content": content_str})
    if system_content and not cleaned:
        cleaned.append({"role": "user", "content": system_content})
    return cleaned

# RecurrentGemma Streamer
def generate_stream_recurrentgemma(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    repeat_context: int,
):
    global model, tokenizer
    from recurrentgemma.model import DecoderCache as RGDecoderCache

    cleaned = clean_messages_for_gemma(messages)
    formatted_prompt = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(formatted_prompt))[None, :]
    
    cache = RGDecoderCache(
        num_layers=model.config.num_hidden_layers,
        lru_width=model.config.lru_width,
        hidden_size=model.config.hidden_size
    )
    
    logits_processors = []
    if repeat_penalty != 1.0:
        logits_processors.append(make_repetition_penalty(repeat_penalty, repeat_context))
        
    def sample_token(logits, generated_tokens):
        if logits_processors and len(generated_tokens) > 0:
            tokens_arr = mx.array(generated_tokens)
            logits = logits[None, :]
            for processor in logits_processors:
                logits = processor(tokens_arr, logits)
            logits = logits[0]
            
        if temp == 0.0:
            return mx.argmax(logits, axis=-1).item()
            
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        if min_p > 0.0:
            logprobs = apply_min_p(logprobs, min_p)
        if top_p > 0.0:
            logprobs = apply_top_p(logprobs, top_p)
            
        return mx.random.categorical(logprobs / temp).item()

    # Prefill
    logits = model(input_ids, offset=0, cache=cache)
    token_logits = logits[:, -1, :]
    token = sample_token(token_logits[0], [])
    
    prefill_tokens = input_ids.shape[1]
    generated_tokens = [token]
    yield tokenizer.decode([token])
    
    # Decode
    offset = prefill_tokens
    for i in range(1, max_tokens):
        curr_input = mx.array([[token]])
        position_ids = mx.array([[offset]])
        
        logits = model(curr_input, position_ids=position_ids, offset=offset, cache=cache)
        token_logits = logits[:, -1, :]
        token = sample_token(token_logits[0], generated_tokens)
            
        generated_tokens.append(token)
        yield tokenizer.decode([token])
        
        if token == tokenizer.eos_token_id:
            break
        offset += 1

# T5v2 Streamer
def generate_stream_t5(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    repeat_context: int,
):
    global model, tokenizer
    from t5v2.model import DecoderCache as T5DecoderCache

    cleaned = clean_messages_for_gemma(messages)
    prompt = cleaned[-1]["content"]
    
    inputs = tokenizer(text=prompt, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])
    
    encoder_outputs = model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
    )
    
    bos_id = model.config.get("bos_token_id", 2)
    eos_id = model.config.get("eos_token_id", 1)
    decoder_input_ids = mx.array([[bos_id]])
    
    num_decoder_layers = model.config["decoder"]["num_hidden_layers"]
    cache = T5DecoderCache(num_decoder_layers)
    
    logits_processors = []
    if repeat_penalty != 1.0:
        logits_processors.append(make_repetition_penalty(repeat_penalty, repeat_context))
        
    def sample_token(logits, generated_tokens):
        if logits_processors and len(generated_tokens) > 0:
            tokens_arr = mx.array(generated_tokens)
            logits = logits[None, :]
            for processor in logits_processors:
                logits = processor(tokens_arr, logits)
            logits = logits[0]
            
        if temp == 0.0:
            return mx.argmax(logits, axis=-1).item()
            
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        if min_p > 0.0:
            logprobs = apply_min_p(logprobs, min_p)
        if top_p > 0.0:
            logprobs = apply_top_p(logprobs, top_p)
            
        return mx.random.categorical(logprobs / temp).item()

    # Prefill
    logits, _ = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        past_key_values=cache,
        encoder_outputs=encoder_outputs,
    )
    logits = logits[:, -1, :]
    token = sample_token(logits[0], [])
    generated_tokens = [token]
    yield tokenizer.decode([token])
    
    # Decode
    decoder_input_ids = mx.array([[token]])
    for token_idx in range(1, max_tokens):
        if token == eos_id:
            break
            
        logits, _ = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            pixel_values=None,
            past_key_values=cache,
            encoder_outputs=encoder_outputs,
        )
        logits = logits[:, -1, :]
        token = sample_token(logits[0], generated_tokens)
        
        if token == eos_id:
            break
            
        generated_tokens.append(token)
        yield tokenizer.decode([token])
        decoder_input_ids = mx.array([[token]])

# Standard MLX-LM Streamer
def generate_stream_standard(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    repeat_context: int,
    enable_thinking: Optional[bool] = None,
):
    global model, tokenizer, processor, draft_model, draft_kind, draft_block_size

    # Extract images from the message content if any exist (OpenAI vision format)
    images = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    img_url = block.get("image_url", {}).get("url")
                    if img_url:
                        images.append(img_url)

    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    # Format the prompt using mlx_vlm's apply_chat_template helper
    try:
        import mlx_vlm
        # Convert model config to a dict format for mlx_vlm.apply_chat_template
        config_dict = model.config
        if not isinstance(config_dict, dict):
            if hasattr(config_dict, "__dict__"):
                config_dict = config_dict.__dict__
            elif hasattr(config_dict, "to_dict"):
                config_dict = config_dict.to_dict()
            else:
                config_dict = {}

        formatted_prompt = mlx_vlm.apply_chat_template(
            processor or tokenizer,
            config_dict,
            messages,
            add_generation_prompt=True,
            num_images=len(images),
            **kwargs
        )
    except Exception as e:
        print(f"Chat template formatting failed: {e}. Falling back to standard apply_chat_template.")
        # Fallback to model's default chat template or simple text template
        # Try without extra kwargs first
        try:
            formatted_prompt = (processor or tokenizer).apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            cleaned = clean_messages_for_gemma(messages)
            if hasattr(processor or tokenizer, "apply_chat_template"):
                formatted_prompt = (processor or tokenizer).apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = cleaned[-1]["content"]

    # Set up generation parameters for mlx_vlm.stream_generate
    gen_args = {
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repeat_penalty if repeat_penalty != 1.0 else None,
        "repetition_context_size": repeat_context,
        **kwargs
    }

    if images:
        gen_args["image"] = images[0] if len(images) == 1 else images

    if draft_model is not None:
        gen_args["draft_model"] = draft_model
        gen_args["draft_kind"] = draft_kind
        if draft_block_size is not None:
            gen_args["draft_block_size"] = draft_block_size

    import mlx_vlm
    # Call stream_generate
    for result in mlx_vlm.stream_generate(model, processor or tokenizer, formatted_prompt, **gen_args):
        # mlx_vlm.stream_generate yields GenerationResult objects
        yield result.text

    # Print speculative decoding statistics to the server console if active
    if draft_model is not None:
        try:
            from mlx_vlm.speculative.utils import format_speculative_stats
            stats = format_speculative_stats(draft_model)
            if stats:
                print(f"\n[Speculative Stats] {stats}\n")
        except Exception as e:
            print(f"Failed to format speculative stats: {e}")

@app.post("/v1/model/unload")
async def unload_model():
    global model, tokenizer, processor, draft_model, draft_kind, draft_block_size, model_type, model_name, loaded_paths
    if model is None:
        return {"status": "ignored", "message": "No model was loaded."}
        
    print(f"Unloading model '{model_name}' and releasing VRAM...")
    
    model = None
    tokenizer = None
    processor = None
    draft_model = None
    draft_kind = None
    draft_block_size = None
    model_type = None
    model_name = "None"
    loaded_paths = {}
    
    # Force reload of local modules to prevent collisions/import issues
    conflicting = [
        "model", "generate", "quantize", "train", "convert",
        "recurrentgemma.generate", "recurrentgemma.model", "recurrentgemma.quantize",
        "t5v2.generate", "t5v2.model", "t5v2.train", "t5v2.convert"
    ]
    for mod in conflicting:
        sys.modules.pop(mod, None)
        
    gc.collect()
    mx.metal.clear_cache()
    
    return {"status": "success", "message": "Model successfully unloaded and VRAM cleared."}

class MCPServerClient:
    def __init__(self, name, command, args, env=None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.process = None
        self.read_thread = None
        self.request_id = 1
        self.pending_requests = {}
        self.tools = []
        self.connected = False
        self.error = None

    def start(self):
        try:
            # Prepare env
            run_env = os.environ.copy()
            if self.env:
                run_env.update(self.env)
                
            # Start process
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
                bufsize=0
            )
            
            self.connected = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            # Handshake
            self.initialize()
            self.fetch_tools()
        except Exception as e:
            self.connected = False
            self.error = str(e)
            print(f"Error starting MCP server {self.name}: {e}")

    def _read_loop(self):
        try:
            for line in iter(self.process.stdout.readline, b''):
                if not line:
                    break
                try:
                    msg = json.loads(line.decode('utf-8'))
                    msg_id = msg.get('id')
                    if msg_id is not None:
                        store = self.pending_requests.get(msg_id)
                        if store:
                            store['response'] = msg
                            store['event'].set()
                except Exception as e:
                    pass
        except Exception as e:
            print(f"Read loop exception for {self.name}: {e}")
        finally:
            self.connected = False

    def send_request(self, method, params, timeout=5):
        if not self.connected or not self.process:
            return {"error": "Server not connected"}
            
        current_id = self.request_id
        self.request_id += 1
        
        event = threading.Event()
        req_store = {"event": event, "response": None}
        self.pending_requests[current_id] = req_store
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": current_id
        }
        
        try:
            raw = json.dumps(payload) + "\n"
            self.process.stdin.write(raw.encode('utf-8'))
            self.process.stdin.flush()
        except Exception as e:
            self.pending_requests.pop(current_id, None)
            return {"error": f"Write failed: {e}"}
            
        success = event.wait(timeout=timeout)
        self.pending_requests.pop(current_id, None)
        
        if not success:
            return {"error": "Request timed out"}
            
        return req_store["response"]

    def initialize(self):
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "unified-mlx-client", "version": "1.0.0"}
        }
        res = self.send_request("initialize", params)
        if res and "error" not in res:
            # Send initialized notification
            notify = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            try:
                self.process.stdin.write((json.dumps(notify) + "\n").encode('utf-8'))
                self.process.stdin.flush()
            except Exception:
                pass

    def fetch_tools(self):
        res = self.send_request("tools/list", {})
        if res and "result" in res and "tools" in res["result"]:
            self.tools = res["result"]["tools"]
        else:
            self.tools = []

    def call_tool(self, tool_name, arguments):
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        return self.send_request("tools/call", params, timeout=30)

    def stop(self):
        self.connected = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


class SSEMCPServerClient:
    def __init__(self, name, url, headers=None):
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.post_url = None
        self.request_id = 1
        self.pending_requests = {}
        self.tools = []
        self.connected = False
        self.error = None
        self.read_thread = None
        self._stop_event = threading.Event()

    def start(self):
        try:
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            # Wait for endpoint to be received
            start_time = time.time()
            while not self.post_url and time.time() - start_time < 5:
                time.sleep(0.1)
                
            if not self.post_url:
                raise Exception("Did not receive endpoint from SSE stream in time.")
                
            self.connected = True
            self.initialize()
            self.fetch_tools()
        except Exception as e:
            self.connected = False
            self.error = str(e)
            # Silent fallback; error is caught and server is skipped

    def _read_loop(self):
        try:
            req_headers = self.headers.copy()
            req_headers["Accept"] = "text/event-stream"
            with requests.get(self.url, headers=req_headers, stream=True, timeout=10) as resp:
                resp.raise_for_status()
                
                event_type = None
                for line in resp.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if line is None:
                        continue
                    if line == "":
                        event_type = None
                        continue
                        
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        data = line[6:]
                        if event_type == "endpoint":
                            self.post_url = urllib.parse.urljoin(self.url, data)
                        elif event_type == "message":
                            try:
                                msg = json.loads(data)
                                msg_id = msg.get('id')
                                if msg_id is not None:
                                    store = self.pending_requests.get(msg_id)
                                    if store:
                                        store['response'] = msg
                                        store['event'].set()
                            except Exception:
                                pass
        except Exception as e:
            # Silently catch read loop exceptions to allow clean skips
            pass
        finally:
            self.connected = False

    def send_request(self, method, params, timeout=10):
        if not self.connected or not self.post_url:
            return {"error": "Server not connected or missing endpoint"}
            
        current_id = self.request_id
        self.request_id += 1
        
        event = threading.Event()
        req_store = {"event": event, "response": None}
        self.pending_requests[current_id] = req_store
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": current_id
        }
        
        try:
            resp = requests.post(self.post_url, json=payload, headers=self.headers, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            self.pending_requests.pop(current_id, None)
            return {"error": f"POST failed: {e}"}
            
        success = event.wait(timeout=timeout)
        self.pending_requests.pop(current_id, None)
        
        if not success:
            return {"error": "Request timed out"}
            
        return req_store["response"]

    def initialize(self):
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "unified-mlx-client", "version": "1.0.0"}
        }
        res = self.send_request("initialize", params)
        if res and "error" not in res:
            # Send initialized notification
            notify = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            try:
                requests.post(self.post_url, json=notify, headers=self.headers, timeout=5)
            except Exception:
                pass

    def fetch_tools(self):
        res = self.send_request("tools/list", {})
        if res and "result" in res and "tools" in res["result"]:
            self.tools = res["result"]["tools"]
        else:
            self.tools = []

    def call_tool(self, tool_name, arguments):
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        return self.send_request("tools/call", params, timeout=30)

    def stop(self):
        self.connected = False
        self._stop_event.set()

def sync_mcp_servers():
    global mcp_clients
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    if not os.path.exists(config_path):
        default_config = {
            "mcpServers": {
                "mcp-duckduckgo": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-duckduckgo"],
                    "enabled": False
                },
                "mcp-filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/bran/src/play/llm"],
                    "enabled": False
                }
            }
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
            
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        def expand_env_vars(obj):
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            elif isinstance(obj, list):
                return [expand_env_vars(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: expand_env_vars(v) for k, v in obj.items()}
            return obj
            
        config = expand_env_vars(config)
    except Exception as e:
        print(f"Error reading mcp.json: {e}")
        return
        
    servers = config.get("mcpServers", {})
    
    # 1. Stop disabled/removed servers
    for name in list(mcp_clients.keys()):
        if name not in servers or not servers[name].get("enabled", False):
            print(f"Stopping MCP server: {name}")
            mcp_clients[name].stop()
            del mcp_clients[name]
            
    # 2. Start newly enabled servers
    for name, cfg in servers.items():
        if cfg.get("enabled", False) and name not in mcp_clients:
            print(f"Starting MCP server: {name}")
            client = None
            if "url" in cfg:
                # SSE transport
                client = SSEMCPServerClient(name, cfg["url"], cfg.get("headers", {}))
            elif "command" in cfg:
                # Stdio transport
                client = MCPServerClient(name, cfg["command"], cfg.get("args", []), cfg.get("env"))
            else:
                print(f"Skipping MCP server '{name}': Neither 'command' nor 'url' found in config.")
                continue

            if client:
                client.start()
                if client.connected:
                    mcp_clients[name] = client
                else:
                    print(f"Skipping MCP server '{name}': Server is not accessible right now.")

def get_active_tools_definitions():
    global mcp_clients
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    disabled_tools_map = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                for name, s_cfg in cfg.get("mcpServers", {}).items():
                    disabled_tools_map[name] = s_cfg.get("disabled_tools", [])
        except Exception:
            pass
            
    definitions = []
    for name, client in mcp_clients.items():
        if client.connected:
            disabled = disabled_tools_map.get(name, [])
            for t in client.tools:
                t_name = t.get("name")
                if t_name not in disabled:
                    desc = t.get("description", "No description")
                    schema = json.dumps(t.get("inputSchema", {}))
                    definitions.append(f"- {t_name}: {desc}. Arguments schema: {schema}")
                    
    return "\n".join(definitions) if definitions else ""

def find_fallback_tool_name(args):
    global mcp_clients
    active_tools = []
    for name, client in mcp_clients.items():
        if client.connected:
            disabled_tools = []
            config_path = os.path.join(workspace_root, "config", "mcp.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    disabled_tools = cfg.get("mcpServers", {}).get(name, {}).get("disabled_tools", [])
                except Exception:
                    pass
            for t in client.tools:
                t_name = t.get("name")
                if t_name not in disabled_tools:
                    active_tools.append(t_name)
                    
    if not active_tools:
        return "duckduckgo_search"
        
    if "query" in args:
        for t_name in active_tools:
            if "search" in t_name.lower():
                return t_name
                
    return active_tools[0]

def get_tool_permission(tool_name):
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    if not os.path.exists(config_path):
        return "ask"
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        for name, s_cfg in cfg.get("mcpServers", {}).items():
            client = mcp_clients.get(name)
            if client and client.connected:
                for t in client.tools:
                    if t.get("name") == tool_name:
                        allowed = s_cfg.get("allowed_tools", [])
                        if tool_name in allowed:
                            return "allow"
                        return "ask"
    except Exception:
        pass
    return "ask"

def parse_tool_call(text):
    # 1. Check XML format: <tool_call>...</tool_call>
    xml_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if xml_match:
        try:
            data = json.loads(xml_match.group(1).strip())
            if "name" in data:
                return data["name"], data.get("arguments", {})
        except Exception:
            pass

    # 2. Check XML format: <call name="tool_name">...</call>
    call_match = re.search(r'<call name="(.*?)">(.*?)</call>', text, re.DOTALL)
    if call_match:
        try:
            args = json.loads(call_match.group(2).strip())
            return call_match.group(1), args
        except Exception:
            return call_match.group(1), {}

    # 3. Check ReAct format: Action: ... and Action Input: ...
    action_match = re.search(r'Action:\s*([a-zA-Z0-9_\-/]+)', text)
    if action_match:
        action = action_match.group(1).strip()
        action_input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)
        if action_input_match:
            try:
                args = json.loads(action_input_match.group(1).strip())
                return action, args
            except Exception:
                pass
        return action, {}

    # 4. Check Markdown JSON code block format
    json_block = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block:
        try:
            data = json.loads(json_block.group(1).strip())
            if "name" in data:
                return data["name"], data.get("arguments", {})
        except Exception:
            pass

    # 5. Check Gemma 4 native format: <|tool_call>call:tool_name{arguments}<tool_call|> or similar
    gemma_match = re.search(r'<\|tool_call>\s*call:?\s*([a-zA-Z0-9_\-/]+)?\s*(\{.*?\})\s*(?:<tool_call\|>|<\|tool_call\|>)?', text, re.DOTALL)
    if not gemma_match:
        # Also try with pipe in opening tag just in case
        gemma_match = re.search(r'<\|tool_call\|>\s*call:?\s*([a-zA-Z0-9_\-/]+)?\s*(\{.*?\})\s*(?:<tool_call\|>|<\|tool_call\|>)?', text, re.DOTALL)
    if not gemma_match:
        # Check without closing tag if it is still generating
        gemma_match = re.search(r'<\|tool_call>\s*call:?\s*([a-zA-Z0-9_\-/]+)?\s*(\{.*?\}$)', text, re.DOTALL)
        
    if gemma_match:
        t_name = gemma_match.group(1)
        args_str = gemma_match.group(2)
        try:
            args_str = args_str.replace('<|\\"[|>', '"').replace('<|\\"|>', '"')
            args = json.loads(args_str.strip())
            
            # If the JSON itself contains name/arguments structure
            if isinstance(args, dict) and "name" in args:
                if not t_name:
                    t_name = args["name"]
                args = args.get("arguments", {})
                
            if not t_name:
                t_name = find_fallback_tool_name(args)
                
            return t_name, args
        except Exception:
            if t_name:
                return t_name, {}

    return None, None

def resolve_canonical_tool_name(tool_name):
    global mcp_clients
    # 1. Exact match
    for name, client in mcp_clients.items():
        if client.connected:
            for t in client.tools:
                if t.get("name") == tool_name:
                    return tool_name
                    
    # 2. Fuzzy match
    for name, client in mcp_clients.items():
        if client.connected:
            for t in client.tools:
                t_name = t.get("name")
                if tool_name.lower() in t_name.lower() or t_name.lower() in tool_name.lower():
                    return t_name
                    
    return None

def execute_mcp_tool(tool_name, arguments):
    global mcp_clients
    for name, client in mcp_clients.items():
        if client.connected:
            for t in client.tools:
                if t.get("name") == tool_name:
                    print(f"Executing MCP tool {tool_name} on {name}")
                    res = client.call_tool(tool_name, arguments)
                    if res and "result" in res:
                        content = res["result"].get("content", [])
                        texts = []
                        for item in content:
                            if item.get("type") == "text":
                                texts.append(item.get("text", ""))
                        return "\n".join(texts) if texts else json.dumps(res["result"])
                    elif res and "error" in res:
                        return f"Error executing tool: {res['error']}"
                    return f"Tool execution returned: {json.dumps(res)}"
    return f"Error: Tool '{tool_name}' not found or its MCP server is offline."

class McpToggleRequest(BaseModel):
    name: str
    enabled: bool

class McpToolToggleRequest(BaseModel):
    server_name: str
    tool_name: str
    enabled: bool

class McpToolPermissionRequest(BaseModel):
    server_name: str
    tool_name: str
    permission: str # "allow" or "ask"

class McpApproveRequest(BaseModel):
    approval_id: str
    decision: str # "allow" or "reject"


class McpInstallRequest(BaseModel):
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None

@app.on_event("startup")
async def startup_event():
    sync_mcp_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global mcp_clients
    print("Shutting down MCP servers...")
    for name, client in list(mcp_clients.items()):
        client.stop()
    mcp_clients.clear()

@app.get("/v1/mcp/servers")
async def get_mcp_servers():
    sync_mcp_servers()
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    if not os.path.exists(config_path):
        return []
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return []
        
    servers = cfg.get("mcpServers", {})
    response = []
    for name, s_cfg in servers.items():
        client = mcp_clients.get(name)
        response.append({
            "name": name,
            "command": s_cfg.get("command", ""),
            "args": s_cfg.get("args", []),
            "enabled": s_cfg.get("enabled", False),
            "connected": client.connected if client else False,
            "error": client.error if client else None,
            "disabled_tools": s_cfg.get("disabled_tools", []),
            "allowed_tools": s_cfg.get("allowed_tools", []),
            "tools": client.tools if client and client.connected else []
        })
    return response

@app.post("/v1/mcp/servers/toggle")
async def toggle_mcp_server(req: McpToggleRequest):
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="mcp.json not found")
        
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")
        
    servers = cfg.get("mcpServers", {})
    if req.name not in servers:
        raise HTTPException(status_code=404, detail=f"Server {req.name} not found in config")
        
    servers[req.name]["enabled"] = req.enabled
    
    try:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
        
    sync_mcp_servers()
    return {"status": "success"}

@app.post("/v1/mcp/servers/tools/toggle")
async def toggle_mcp_tool_endpoint(req: McpToolToggleRequest):
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="mcp.json not found")
        
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")
        
    servers = cfg.get("mcpServers", {})
    if req.server_name not in servers:
        raise HTTPException(status_code=404, detail=f"Server {req.server_name} not found in config")
        
    s_cfg = servers[req.server_name]
    disabled = s_cfg.get("disabled_tools", [])
    
    if req.enabled:
        if req.tool_name in disabled:
            disabled.remove(req.tool_name)
    else:
        if req.tool_name not in disabled:
            disabled.append(req.tool_name)
            
    s_cfg["disabled_tools"] = disabled
    
    try:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
        
    return {"status": "success"}

@app.post("/v1/mcp/servers/tools/permission")
async def toggle_mcp_tool_permission(req: McpToolPermissionRequest):
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="mcp.json not found")
        
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")
        
    servers = cfg.get("mcpServers", {})
    if req.server_name not in servers:
        raise HTTPException(status_code=404, detail=f"Server {req.server_name} not found in config")
        
    s_cfg = servers[req.server_name]
    allowed = s_cfg.get("allowed_tools", [])
    
    if req.permission == "allow":
        if req.tool_name not in allowed:
            allowed.append(req.tool_name)
    else:
        if req.tool_name in allowed:
            allowed.remove(req.tool_name)
            
    s_cfg["allowed_tools"] = allowed
    
    try:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
        
    return {"status": "success"}

@app.post("/v1/mcp/approve")
async def approve_mcp_tool(req: McpApproveRequest):
    global pending_approvals
    if req.approval_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found or expired")
        
    pending_approvals[req.approval_id]["decision"] = req.decision
    pending_approvals[req.approval_id]["event"].set()
    return {"status": "success"}


@app.post("/v1/mcp/install")
async def install_mcp_server(req: McpInstallRequest):
    config_path = os.path.join(workspace_root, "config", "mcp.json")
    
    cfg = {"mcpServers": {}}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass
            
    servers = cfg.get("mcpServers", {})
    servers[req.name] = {
        "command": req.command,
        "args": req.args,
        "enabled": False,
        "disabled_tools": []
    }
    if req.env:
        servers[req.name]["env"] = req.env
        
    try:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
        
    sync_mcp_servers()
    return {"status": "success"}

@app.post("/v1/model/load")
async def load_model_endpoint(request: ModelLoadRequest):
    global model, tokenizer, processor, draft_model, draft_kind, draft_block_size, model_type, model_name, loaded_paths, ACTIVE_IGNORE_LAYERS
    
    # Automatically unload first
    await unload_model()
    
    # Set the ignore layers for the monkeypatch before MLX loads weights
    ACTIVE_IGNORE_LAYERS = request.ignore_layers or []
    
    t_start = time.perf_counter()
    m_type = request.model_type.lower()
    
    print(f"Loading model type '{m_type}' from paths...")
    
    try:
        # Clear modules before importing
        conflicting = [
            "model", "generate", "quantize", "train", "convert",
            "recurrentgemma.generate", "recurrentgemma.model", "recurrentgemma.quantize",
            "t5v2.generate", "t5v2.model", "t5v2.train", "t5v2.convert"
        ]
        for mod in conflicting:
            sys.modules.pop(mod, None)

        if m_type == "recurrentgemma":
            rg_dir = os.path.join(workspace_root, "engines", "recurrentgemma")
            original_path = list(sys.path)
            sys.path.insert(0, rg_dir)
            try:
                from recurrentgemma.generate import load_model as load_rg
                model, _ = load_rg(
                    request.config_path, 
                    request.weights_path, 
                    attention_window=request.attention_window
                )
            finally:
                sys.path = original_path
            tokenizer = AutoTokenizer.from_pretrained(request.tokenizer_path)
            model_name = os.path.basename(os.path.dirname(request.config_path))
            
        elif m_type == "t5v2":
            t5v2_dir = os.path.join(workspace_root, "engines", "t5v2")
            original_path = list(sys.path)
            sys.path.insert(0, t5v2_dir)
            try:
                from t5v2.generate import load_model as load_t5
                model, _ = load_t5(
                    request.config_path, 
                    request.weights_path, 
                    adapter_path=request.adapter_path
                )
            finally:
                sys.path = original_path
            tokenizer = AutoProcessor.from_pretrained(request.tokenizer_path)
            model_name = os.path.basename(os.path.dirname(request.config_path))
            
        elif m_type == "standard":
            tokenizer_path = request.tokenizer_path
            if not os.path.isabs(tokenizer_path):
                tokenizer_path = os.path.join(workspace_root, tokenizer_path)
            
            adapter_path = request.adapter_path
            if adapter_path and not os.path.isabs(adapter_path):
                adapter_path = os.path.join(workspace_root, adapter_path)
                
            # Temporary monkey patches to dynamically ignore layers and save VRAM
            import mlx.nn as nn
            import mlx_vlm.utils
            from mlx.utils import tree_flatten
            
            original_update_module_configs = mlx_vlm.utils.update_module_configs
            original_load_weights = nn.Module.load_weights
            
            def custom_update_module_configs(model_config, model_class, config, modules):
                res = original_update_module_configs(model_config, model_class, config, modules)
                global ACTIVE_IGNORE_LAYERS
                if ACTIVE_IGNORE_LAYERS:
                    is_text_only = any("vision" in lay.lower() or "projector" in lay.lower() or "audio" in lay.lower() for lay in ACTIVE_IGNORE_LAYERS)
                    if is_text_only:
                        print("Text-only mode: Overriding module configs to None to prevent model instantiation and VRAM allocation.")
                        res.vision_config = None
                        res.audio_config = None
                        if hasattr(res, "perceiver_config"):
                            res.perceiver_config = None
                        if hasattr(res, "projector_config"):
                            res.projector_config = None
                return res

            def custom_load_weights(self, weights, *args, **l_kwargs):
                model_keys = set(k for k, _ in tree_flatten(self.parameters()))
                filtered_weights = [(k, v) for k, v in weights if k in model_keys]
                dropped_keys = [k for k, _ in weights if k not in model_keys]
                if dropped_keys:
                    print(f"[Text-Only Loader] Bypassed loading {len(dropped_keys)} multimodal weight keys (saved VRAM).")
                return original_load_weights(self, filtered_weights, *args, **l_kwargs)
                
            mlx_vlm.utils.update_module_configs = custom_update_module_configs
            nn.Module.load_weights = custom_load_weights
            
            try:
                from mlx_vlm import load as load_std
                model, processor = load_std(
                    tokenizer_path,
                    adapter_path=adapter_path
                )
            finally:
                # Restore original loaders
                mlx_vlm.utils.update_module_configs = original_update_module_configs
                nn.Module.load_weights = original_load_weights
                
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            model_name = os.path.basename(tokenizer_path)

            if request.draft_model_path:
                draft_path = request.draft_model_path
                if not os.path.isabs(draft_path):
                    draft_path = os.path.join(workspace_root, draft_path)
                print(f"Loading draft model ({request.draft_kind or 'auto'}) from {draft_path}...")
                from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
                draft_model, resolved_kind = load_drafter(draft_path, kind=request.draft_kind)
                validate_drafter_compatibility(model, draft_model, resolved_kind)
                draft_kind = resolved_kind
                draft_block_size = request.draft_block_size or 4
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")
            
        # Check if chat_template_path is provided and load it
        if request.chat_template_path:
            template_path = request.chat_template_path
            # Resolve relative paths relative to workspace_root
            if not os.path.isabs(template_path):
                template_path = os.path.join(workspace_root, template_path)
            
            if os.path.exists(template_path):
                print(f"Loading custom chat template from: {template_path}")
                with open(template_path, "r", encoding="utf-8") as f:
                    template_str = f.read()
                
                # Check if we have loaded a tokenizer or processor
                if tokenizer is not None:
                    tokenizer.chat_template = template_str
            else:
                print(f"Warning: Custom chat template path not found: {template_path}")
            
        model_type = m_type
        loaded_paths = {
            "config_path": request.config_path,
            "weights_path": request.weights_path,
            "tokenizer_path": request.tokenizer_path,
            "adapter_path": request.adapter_path,
            "attention_window": request.attention_window,
            "chat_template_path": request.chat_template_path,
            "draft_model_path": request.draft_model_path,
            "draft_kind": request.draft_kind,
            "draft_block_size": request.draft_block_size
        }
        
        load_time = time.perf_counter() - t_start
        print(f"Successfully loaded '{model_name}' in {load_time:.2f}s!")
        return {
            "status": "success",
            "model_name": model_name,
            "model_type": model_type,
            "load_time_seconds": load_time
        }
        
    except Exception as e:
        # Revert state on error
        await unload_model()
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/model/status")
async def model_status():
    global model, draft_model, draft_kind, draft_block_size, model_type, model_name, loaded_paths
    
    # Calculate memory stats
    active_mem = 0.0
    cache_mem = 0.0
    peak_mem = 0.0
    gpu_limit = 0.0
    
    try:
        active_mem = mx.metal.get_active_memory() / (1024**3)
        cache_mem = mx.metal.get_cache_memory() / (1024**3)
        peak_mem = mx.metal.get_peak_memory() / (1024**3)
    except Exception:
        try:
            active_mem = mx.get_active_memory() / (1024**3)
            peak_mem = mx.get_peak_memory() / (1024**3)
        except Exception:
            pass

    try:
        info = mx.device_info()
        gpu_limit = info.get('max_recommended_working_set_size', 0) / (1024**3)
        if gpu_limit == 0 and psutil is not None:
            gpu_limit = (psutil.virtual_memory().total / (1024**3)) * 0.75
    except Exception:
        try:
            if psutil is not None:
                gpu_limit = (psutil.virtual_memory().total / (1024**3)) * 0.75
        except Exception:
            pass

    # Also check system memory using psutil if available
    system_ram_used = None
    system_ram_total = None
    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            system_ram_used = vm.used / (1024**3)
            system_ram_total = vm.total / (1024**3)
    except Exception:
        pass

    return {
        "loaded": model is not None,
        "model_name": model_name,
        "model_type": model_type,
        "paths": loaded_paths,
        "active_mem_gb": round(active_mem, 2),
        "cache_mem_gb": round(cache_mem, 2),
        "peak_mem_gb": round(peak_mem, 2),
        "gpu_limit_gb": round(gpu_limit, 2),
        "system_ram_used_gb": round(system_ram_used, 2) if system_ram_used is not None else None,
        "system_ram_total_gb": round(system_ram_total, 2) if system_ram_total is not None else None,
        "tool_calling_supported": model is not None and model_type == "standard",
        "speculative_decoding": draft_model is not None,
        "draft_kind": draft_kind,
        "draft_block_size": draft_block_size
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, tokenizer, model_type
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="No model is currently loaded. Load a model first via /v1/model/load.")
        
    messages_dict = [m.model_dump() for m in request.messages]
    
    # Select generator driver based on loaded model type
    if model_type == "recurrentgemma":
        stream_fn = generate_stream_recurrentgemma
    elif model_type == "t5v2":
        stream_fn = generate_stream_t5
    else:
        stream_fn = generate_stream_standard
        
    gen_kwargs = {
        "messages": messages_dict,
        "max_tokens": request.max_tokens,
        "temp": request.temperature,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "min_p": request.min_p,
        "repeat_penalty": request.repeat_penalty,
        "repeat_context": request.repeat_context,
    }
    if stream_fn == generate_stream_standard:
        gen_kwargs["enable_thinking"] = request.enable_thinking

    if request.stream:
        async def event_generator():
            created_time = int(time.time())
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(0)
            
            curr_messages = list(messages_dict)
            active_tools = get_active_tools_definitions()
            if active_tools and model_type == "standard":
                sys_msg = None
                for msg in curr_messages:
                    if msg["role"] == "system":
                        sys_msg = msg
                        break
                tool_prompt = f"\n\nYou have access to the following tools via MCP servers:\n{active_tools}\n\nTo call a tool, you MUST reply in one of these formats:\n\nFormat 1 (XML):\n<tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"val1\"}}}}\n</tool_call>\n\nFormat 2 (ReAct):\nAction: tool_name\nAction Input: {{\"arg1\": \"val1\"}}\n\nDo not output anything else when calling a tool. Once you receive the tool response (Observation), you can formulate your final answer."
                if sys_msg:
                    sys_msg["content"] += tool_prompt
                else:
                    curr_messages.insert(0, {"role": "system", "content": tool_prompt})

            loop_count = 0
            while loop_count < 5:
                loop_count += 1
                gen_kwargs["messages"] = curr_messages
                accumulated_text = ""
                
                for token_text in stream_fn(**gen_kwargs):
                    accumulated_text += token_text
                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}]})}\n\n"
                    await asyncio.sleep(0)
                    
                    if "<tool_call|>" in accumulated_text or "</tool_call>" in accumulated_text or "<|tool_call|>" in accumulated_text:
                        break
                    if "<|tool_response>" in accumulated_text:
                        accumulated_text = accumulated_text.split("<|tool_response>")[0]
                        break
                    if "\nObservation:" in accumulated_text:
                        accumulated_text = accumulated_text.split("\nObservation:")[0]
                        break
                    
                tool_name, tool_args = parse_tool_call(accumulated_text)
                if tool_name and model_type == "standard":
                    canonical = resolve_canonical_tool_name(tool_name)
                    if canonical:
                        tool_name = canonical
                    
                    # Check permission
                    permission = get_tool_permission(tool_name)
                    decision = "allow"
                    if permission == "ask":
                        # We must ask the user
                        approval_id = f"appr-{uuid.uuid4()}"
                        event = asyncio.Event()
                        pending_approvals[approval_id] = {
                            "event": event,
                            "decision": None
                        }
                        # Send the approval request chunk to the client
                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'tool_approval_required': {'approval_id': approval_id, 'tool_name': tool_name, 'arguments': tool_args}}, 'finish_reason': None}]})}\n\n"
                        await asyncio.sleep(0)
                        
                        try:
                            # Wait for client's approval response (timeout 60 seconds)
                            await asyncio.wait_for(event.wait(), timeout=60.0)
                            decision = pending_approvals[approval_id]["decision"]
                        except asyncio.TimeoutError:
                            decision = "reject"
                        finally:
                            pending_approvals.pop(approval_id, None)
                            
                    # If allowed, execute the tool
                    if decision == "allow":
                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': f"\nObservation ({tool_name}): "}, 'finish_reason': None}]})}\n\n"
                        await asyncio.sleep(0)
                        
                        tool_output = execute_mcp_tool(tool_name, tool_args)
                        
                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': f"{tool_output}\n"}, 'finish_reason': None}]})}\n\n"
                        await asyncio.sleep(0)
                    else:
                        tool_output = "Error: Tool execution rejected by the user."
                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': f"\nObservation ({tool_name}): {tool_output}\n"}, 'finish_reason': None}]})}\n\n"
                        await asyncio.sleep(0)
                        
                    curr_messages.append({"role": "assistant", "content": accumulated_text})
                    curr_messages.append({"role": "user", "content": f"Observation ({tool_name}): {tool_output}"})
                else:
                    break
                    
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    else:
        curr_messages = list(messages_dict)
        active_tools = get_active_tools_definitions()
        if active_tools and model_type == "standard":
            sys_msg = None
            for msg in curr_messages:
                if msg["role"] == "system":
                    sys_msg = msg
                    break
            tool_prompt = f"\n\nYou have access to the following tools via MCP servers:\n{active_tools}\n\nTo call a tool, you MUST reply in one of these formats:\n\nFormat 1 (XML):\n<tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"val1\"}}}}\n</tool_call>\n\nFormat 2 (ReAct):\nAction: tool_name\nAction Input: {{\"arg1\": \"val1\"}}\n\nDo not output anything else when calling a tool. Once you receive the tool response (Observation), you can formulate your final answer."
            if sys_msg:
                sys_msg["content"] += tool_prompt
            else:
                curr_messages.insert(0, {"role": "system", "content": tool_prompt})

        loop_count = 0
        total_tokens = 0
        final_text = ""
        while loop_count < 5:
            loop_count += 1
            gen_kwargs["messages"] = curr_messages
            
            tokens = []
            for token_text in stream_fn(**gen_kwargs):
                tokens.append(token_text)
                current_text = "".join(tokens)
                
                if "<tool_call|>" in current_text or "</tool_call>" in current_text or "<|tool_call|>" in current_text:
                    break
                if "<|tool_response>" in current_text:
                    tokens = [current_text.split("<|tool_response>")[0]]
                    break
                if "\nObservation:" in current_text:
                    tokens = [current_text.split("\nObservation:")[0]]
                    break
                
            full_text = "".join(tokens)
            total_tokens += len(tokens)
            
            tool_name, tool_args = parse_tool_call(full_text)
            if tool_name and model_type == "standard":
                canonical = resolve_canonical_tool_name(tool_name)
                if canonical:
                    tool_name = canonical
                permission = get_tool_permission(tool_name)
                if permission == "allow":
                    tool_output = execute_mcp_tool(tool_name, tool_args)
                else:
                    tool_output = "Error: Tool execution requires permission, but this is a non-streaming request. Please enable streaming."
                curr_messages.append({"role": "assistant", "content": full_text})
                curr_messages.append({"role": "user", "content": f"Observation ({tool_name}): {tool_output}"})
            else:
                final_text = full_text
                break
                
        created_time = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }

@app.get("/v1/models/registry")
async def list_models_registry():
    registry_path = os.path.join(workspace_root, "config", "models_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            return json.load(f)
    return {}

@app.get("/v1/presets")
async def list_presets():
    presets_dir = os.path.join(workspace_root, "config", "presets")
    presets = []
    if os.path.exists(presets_dir):
        for f in os.listdir(presets_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(presets_dir, f), "r") as fp:
                        presets.append(json.load(fp))
                except Exception as e:
                    print(f"Failed to load preset {f}: {e}")
    return presets

@app.post("/v1/presets/save")
async def save_preset(payload: dict):
    preset_name = payload.get("preset_name")
    if not preset_name:
        raise HTTPException(status_code=400, detail="preset_name is required")
        
    safe_name = re.sub(r'[^A-Za-z0-9_-]', '-', preset_name)
    filename = f"{safe_name}.json"
    
    presets_dir = os.path.join(workspace_root, "config", "presets")
    os.makedirs(presets_dir, exist_ok=True)
    
    filepath = os.path.join(presets_dir, filename)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
        
    return {"status": "success", "file": filename}

@app.get("/v1/models")
async def list_models():
    global model_name
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Model MLX Gateway API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8089, help="Port to bind")
    
    args = parser.parse_args()
    print(f"\nStarting MLX Gateway Server on http://{args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
