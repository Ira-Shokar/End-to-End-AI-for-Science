"""
LLM Client - Multi-Backend OpenAI Router
Simple, traceable, fail-fast design
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from contextlib import asynccontextmanager
from functools import partial, wraps
from typing import Dict, List
import httpx, os, json, logging, asyncio

from observability import get_observability

logger, tracer, propagator, traced = get_observability("llm-client")

# HTTP client
Client = partial(httpx.AsyncClient, timeout=httpx.Timeout(60))

# Global state - computed once on startup
BACKENDS = {}  # url -> {models: {id -> model_data}, key_env: str}
MODEL_MAP = {}  # model_id -> backend_url
INITIAL_KEYS = {k: os.getenv(k) for k in ["NVIDIA_API_KEY", "OPENAI_API_KEY"]}

def get_missing_model_msg(model):
    return {"error": {
        "message": f"The model \'{model}\' does not exist", 
        "type": "invalid_request_error", 
        "param": "model", 
        "code": "model_not_found"}}

# Backend configs - static, validated on startup
BACKEND_CONFIGS = [
    {
        "url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "exclude": [
            "adept/fuyu-8b", "baai/bge-m3", # "bigcode/starcoder2-15b", "bigcode/starcoder2-7b",
            "google/deplot", "google/gemma-2b", "google/paligemma", "liuhaotian/llava-v1.6-34b",
            "liuhaotian/llava-v1.6-mistral-7b", "microsoft/kosmos-2", "mistralai/mixtral-8x22b-v0.1",
            "nvidia/nemotron-4-340b-reward", "yentinglin/llama-3-taiwan-70b-instruct"
        ],
    },
    {"url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "exclude": []}
]

# @traced("validate_backend")
async def validate_backend(config: Dict) -> bool:
    """
    Validate backend is accessible and has valid auth
    Returns False on permission errors - backend will be ejected
    """
    url, key_env = config["url"], config["key_env"]
    key = os.getenv(key_env)
    
    if not key:
        logger.warning(f"No key for {url} - skipping")
        return False
    
    try:
        async with Client() as c:
            headers = {"authorization": f"Bearer {key}"}
            r = await c.get(f"{url}/models", headers=headers, timeout=10.0)
            
            # Permission errors = permanent failure
            if r.status_code in [401, 403]:
                logger.error(f"Auth failed for {url} - ejecting backend")
                return False
            
            if r.status_code != 200:
                logger.warning(f"Backend {url} returned {r.status_code}")
                return False
            
            # Parse models
            data = r.json().get('data', [])
            models = {m['id']: m for m in data if m['id'] not in config.get('exclude', [])}
            
            # Store validated backend
            BACKENDS[url] = {"models": models, "key_env": key_env}
            MODEL_MAP.update({mid: url for mid in models.keys()})

            if url == "https://integrate.api.nvidia.com/v1":
                FORCE_MODELS = {
                    'nv-rerank-qa-mistral-4b:1': 'https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking',
                    'nvidia/llama-3.2-nv-rerankqa-1b-v2': 'https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking',
                    'nvidia/llama-3.2-nemoretriever-500m-rerank-v2': 'https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nemoretriever-500m-rerank-v2/reranking',
                }
                MODEL_MAP.update(FORCE_MODELS)
                for mid, murl in FORCE_MODELS.items():
                    BACKENDS[murl] = {"models": {mid: {"id": mid}}, "key_env": key_env}
            logger.info(f"✓ {url}: {len(models)} models")
            return True
            
    except Exception as e:
        logger.error(f"Failed to validate {url}: {e}")
        return False

# @traced("discover_backends")
async def discover_backends():
    """Pre-compute all accessible backends - run once on startup"""
    BACKENDS.clear()
    MODEL_MAP.clear()
    
    results = await asyncio.gather(*[validate_backend(cfg) for cfg in BACKEND_CONFIGS])
    valid_count = sum(results)
    
    if valid_count == 0:
        logger.error("No backends available!")
    else:
        logger.info(f"Ready: {valid_count} backends, {len(MODEL_MAP)} models")
    return results

# Lifecycle - clean and simple
@asynccontextmanager
async def lifespan(app: FastAPI):
    await discover_backends()
    yield

app = FastAPI(title="LLM Client", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"backends": len(BACKENDS), "models": len(MODEL_MAP)}

@app.get("/v1/models")
async def list_models():
    """Aggregates models from all validated backends"""
    all_models = [m for b in BACKENDS.values() for m in b["models"].values()]
    return {"object": "list", "data": all_models}

@app.get("/v1/models/{model:path}")
async def get_model(model: str):
    """Single model lookup"""
    if url := MODEL_MAP.get(model):
        return BACKENDS[url]["models"][model]
    raise HTTPException(404, get_missing_model_msg(model))

def merge_openai_chunks(chunks: dict[int, dict]) -> dict:
    content = []
    usage = None
    finish_reason = None
    stop_reason = None
    first = chunks.get(0, {})
    text_class = ""
    acc_choice = {"index": 0}

    for i, c in chunks.items():
        if not c or not c.get("choices"): continue
        choices = c["choices"][0]
        if "delta" in choices:
            content.append(choices.get("delta", {}).get("content", ""))
            text_class = "content"
        elif "text" in choices:
            content.append(choices.get("text", ""))
            text_class = "text"
        usage = c.get("usage", usage)
        if c["choices"][0].get("finish_reason"): 
            acc_choice["finish_reason"] = c["choices"][0]["finish_reason"]
            acc_choice["stop_reason"] = c["choices"][0].get("stop_reason")

    if text_class == "content":
        acc_choice["message"] = {"role": "assistant", "content": "".join(content)}
    elif text_class == "text":
        acc_choice["text"] = "".join(content)

    return {
        "id": first.get("id", "synthetic"),
        "object": first.get("object", "chat.completion"),
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": [acc_choice],
        "usage": usage,
        "num_chunks": len(chunks)
    }


@app.post("/v1/{path:path}")
async def proxy(request: Request, path: str):
    """
    Core proxy logic - simple and traceable:
    1. Parse request body
    2. Lookup backend from MODEL_MAP
    3. Forward request with auth
    4. Stream or return response
    """
    body = await request.body()
    with tracer.start_as_current_span("proxy_request") as span:
        span.set_attribute("path", path)
        try:
            body_json = json.loads(body)
        except Exception as e:
            err_response = dict(status_code=400, detail={"error": f"Malformed JSON {e}"})
            span.set_attribute("error", err_response)
            raise HTTPException(**err_response)

        model = body_json.get("model")
        stream = body_json.get("stream", False)
        span.set_attribute("model", model)
        span.set_attribute("stream", stream)
        
        if not (url := MODEL_MAP.get(model)):
            err_response = dict(status_code=400, detail=get_missing_model_msg(model))
            span.set_attribute("request", json.dumps(body_json))        
            span.set_attribute("error", err_response)
            raise HTTPException(**err_response)

        backend = BACKENDS[url]
        span.set_attribute("backend_url", url)
        key = os.getenv(backend["key_env"])
        span.set_attribute("request", json.dumps(body_json)) 

        # Build proxied request
        headers = {
            "authorization": f"Bearer {key}",
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json"
        }

        payload_url = url if url.endswith("ranking") else f"{url}/{path}"
        
        payload = dict(url=payload_url, content=json.dumps(body_json).encode(), headers=headers)
        # span.set_attribute("payload", str(payload)) 
        
        # Forward request
        if not stream:
            async with Client() as c:
                r = await c.post(**payload)
            span.set_attribute("content", r.content)
            span.set_attribute("status_code", r.status_code)
            return Response(content=r.content, status_code=r.status_code)
        
        # Streaming path
        async def stream_response():
            async with Client().stream("POST", **payload) as r:
                # Check error before committing to stream
                span.set_attribute("status_code", r.status_code)
                if r.status_code != 200:
                    yield r
                    return
                yield r
                async for chunk in r.aiter_bytes():
                    yield chunk

        gen = stream_response()
        first = await gen.__anext__()
        raw_content = await first.aread()
        decoded = raw_content.decode(errors="replace")
        parsed = decoded 
        try:
            # data: {"id":"chat-c9f74386cd2b4bd48f915d818216c5c3","object":"chat.completion.chunk",...}\n\ndata: ...
            # data: [DONE]
            list_spec = decoded[6:].split("\n\ndata: ")[:-1]
            if list_spec:
                parsed_dict = {i: json.loads(c) for i, c in enumerate(list_spec)}
                batch_summary = merge_openai_chunks(parsed_dict)
                parsed = json.dumps(parsed_dict)
                span.set_attribute("first_batch.aggregate", json.dumps(batch_summary))
                span.set_attribute("first_batch.head", json.dumps(parsed_dict[0]))
                span.set_attribute("first_batch.tail", json.dumps(parsed_dict[len(parsed_dict)-1]))
        except Exception as e:
            logger.warning(str(e))
            span.set_attribute("first_batch.body", parsed)
        if first.status_code != 200:
            raise HTTPException(status_code=first.status_code, detail=parsed)
        return StreamingResponse(gen, media_type="text/event-stream")

@app.post("/rediscover")
async def rediscover():
    """Manually trigger backend rediscovery"""
    await discover_backends()
    return {"backends": len(BACKENDS), "models": len(MODEL_MAP)}

if __name__ == "__main__":
    import uvicorn

    DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
    log_level = "debug" if DEBUG_MODE else "info"

    uvicorn.run(
        "llm_client:app",
        host="0.0.0.0",
        port=9000,
        log_level=log_level,
    )
