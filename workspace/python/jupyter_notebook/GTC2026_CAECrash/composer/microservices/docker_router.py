"""
Docker Router - Container Management Microservice
Provides REST API for Docker container lifecycle management
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import docker
from typing import List, Dict, Optional
import logging

# Setup logging for glanceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Docker client - fails fast if Docker unavailable
try:
    client = docker.from_env()
    logger.info("Docker client initialized")
except Exception as e:
    logger.error(f"Docker unavailable: {e}")
    client = None

app = FastAPI(title="Docker Router", description="Container lifecycle management API")

# --- Request Models ---
class ApiKey(BaseModel):
    """API key validation model - enforces NVIDIA key format"""
    nvapi_key: str = Field(..., description="NVIDIA API key starting with 'nvapi-'")
    
    @validator('nvapi_key')
    def validate_key_prefix(cls, v):
        if not v.startswith('nvapi-'): raise ValueError('Key must start with "nvapi-"')
        return v

# Global state - simple in-memory storage
STATE = {"api_key": None}

# --- Core Routes ---
@app.get("/")
async def root():
    """Lists available endpoints with descriptions"""
    return {route.path: route.description or "No description" for route in app.routes if route.path not in ["/openapi.json", "/docs"]}

@app.get("/health")
async def health():
    """Health check - verifies Docker connectivity"""
    if not client: raise HTTPException(500, "Docker unavailable")
    try:
        client.ping()
        return {"status": "healthy", "docker": "connected"}
    except: raise HTTPException(503, "Docker connection failed")

# --- Container Management ---
@app.get("/containers")
async def list_containers(all: bool = True) -> List[Dict]:
    """Lists all containers with id, name, status"""
    if not client: raise HTTPException(500, "Docker unavailable")
    return [{"id": c.id[:12], "name": c.name, "status": c.status} for c in client.containers.list(all=all)]

@app.get("/containers/{name}/logs")
async def get_logs(name: str, tail: int = 100) -> Dict[str, str]:
    """Fetches recent logs from specified container"""
    if not client: raise HTTPException(500, "Docker unavailable")
    try:
        container = client.containers.get(name)
        return {"logs": container.logs(tail=tail).decode('utf-8', errors='ignore')}
    except docker.errors.NotFound: raise HTTPException(404, f"Container '{name}' not found")
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/containers/{name}/restart")
async def restart_container(name: str) -> Dict[str, str]:
    """Restarts specified container"""
    if not client: raise HTTPException(500, "Docker unavailable")
    try:
        container = client.containers.get(name)
        container.restart()
        logger.info(f"Restarted container: {name}")
        return {"status": "restarted", "container": name}
    except docker.errors.NotFound: raise HTTPException(404, f"Container '{name}' not found")
    except Exception as e: raise HTTPException(500, str(e))

# --- Key Management ---
@app.post("/key")
async def set_key(key: ApiKey):
    """Stores API key in memory - validates format"""
    STATE["api_key"] = key.nvapi_key
    logger.info("API key updated")
    return {"status": "key_set"}

# @app.get("/key")
# async def get_key() -> Dict[str, Optional[str]]:
#     """Retrieves stored API key - masked for security"""
#     key = STATE.get("api_key")
#     return {"key": f"{key[:10]}...{key[-4:]}" if key else None}

@app.post("/run_assessment")
async def restart_container(submission: dict):
    """Useful to enable the assessment logic. Will be used in Notebook 4E"""
    from assessment.run_assessment import run_assessment
    return run_assessment(submission)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)