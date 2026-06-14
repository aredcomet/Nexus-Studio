import argparse
import json
import time
import uuid
from typing import Dict, List, Optional, Union
import os
import sys

# Ensure the parent recurrentgemma directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import mlx.core as mx
from transformers import AutoTokenizer

# Import model components and generate helper logic
from model import DecoderCache
from generate import load_model
from mlx_lm.sample_utils import (
    apply_top_k,
    apply_top_p,
    apply_min_p,
    make_repetition_penalty,
)

app = FastAPI(title="RecurrentGemma MLX API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model, config, tokenizer
model = None
config = None
tokenizer = None
model_name = "recurrentgemma-2b-it"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "recurrentgemma-2b-it"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    repeat_penalty: float = 1.1
    repeat_context: int = 20
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

def clean_messages_for_gemma(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merges system messages into the first user message for compatibility with Gemma's template."""
    cleaned = []
    system_content = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_content = content
        elif role == "user":
            if system_content:
                cleaned.append({"role": "user", "content": f"{system_content}\n\n{content}"})
                system_content = ""
            else:
                cleaned.append({"role": "user", "content": content})
        else:
            cleaned.append({"role": role, "content": content})
    # If there was a system prompt but no user prompt, add a dummy user prompt
    if system_content and not cleaned:
        cleaned.append({"role": "user", "content": system_content})
    return cleaned

def generate_stream(
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
    
    cleaned = clean_messages_for_gemma(messages)
    formatted_prompt = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(formatted_prompt))[None, :]
    
    cache = DecoderCache(
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

    # 1. Prefill step
    logits = model(input_ids, offset=0, cache=cache)
    token_logits = logits[:, -1, :]
    token = sample_token(token_logits[0], [])
    
    prefill_tokens = input_ids.shape[1]
    generated_tokens = [token]
    
    yield tokenizer.decode([token])
    
    # 2. Generation step
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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    messages_dict = [m.model_dump() for m in request.messages]
    
    if request.stream:
        async def event_generator():
            created_time = int(time.time())
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            
            # Yield role block first
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Stream tokens
            for token_text in generate_stream(
                messages=messages_dict,
                max_tokens=request.max_tokens,
                temp=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                repeat_penalty=request.repeat_penalty,
                repeat_context=request.repeat_context,
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
            # Yield final chunk
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    else:
        # Non-streaming mode
        tokens = []
        for token_text in generate_stream(
            messages=messages_dict,
            max_tokens=request.max_tokens,
            temp=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=request.min_p,
            repeat_penalty=request.repeat_penalty,
            repeat_context=request.repeat_context,
        ):
            tokens.append(token_text)
            
        full_text = "".join(tokens)
        created_time = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
                "total_tokens": len(tokens)
            }
        }
        return response

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            }
        ]
    }

def main():
    global model, config, tokenizer, model_name
    
    parser = argparse.ArgumentParser(description="OpenAI-Compatible FastAPI server for RecurrentGemma in MLX")
    parser.add_argument("--config", type=str, default="models/recurrentgemma-2b-it/config.json", help="Path to config.json")
    parser.add_argument("--weights", type=str, default="weights/recurrentgemma-2b-it-mxfp4/weights.safetensors", help="Path to weights file")
    parser.add_argument("--tokenizer", type=str, default="models/recurrentgemma-2b-it", help="Path to tokenizer folder")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--attention-window", type=int, default=None, help="Override sliding window attention size")
    
    args = parser.parse_args()
    
    # Check fallback for weights if default doesn't exist
    if not os.path.exists(args.weights):
        alt_weights = "weights/recurrentgemma-2b-it-4bit/weights.safetensors"
        if os.path.exists(alt_weights):
            args.weights = alt_weights
        else:
            alt_weights = "weights/recurrentgemma-2b-it/weights.safetensors"
            if os.path.exists(alt_weights):
                args.weights = alt_weights
            
    # Load model and tokenizer
    model, config = load_model(args.config, args.weights, attention_window=args.attention_window)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_name = os.path.basename(os.path.dirname(args.config))
    
    print(f"\nStarting API Server on http://{args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
