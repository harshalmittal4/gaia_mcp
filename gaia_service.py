import os
import json
import asyncio
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import BaseModel, Field

import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
GAIA_HOST = os.getenv("GAIA_HOST", "https://helios.cohesity.com")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "")
print(GAIA_HOST)
print(API_KEY_HEADER)

# === Pydantic Models ===
class ExecuteParams(BaseModel):
    semantic_search_string: Optional[str] = None
    keyword: Optional[str] = None
    object_types: Optional[List[str]] = Field(default=None, alias="objectTypes")
    file_type: Optional[List[str]] = None
    file_greater_than_kb: Optional[int] = None
    file_less_than_kb: Optional[int] = None

class Document(BaseModel):
    id: str
    text: Optional[str]
    metadata: Dict[str, Any]

class ExecuteResult(BaseModel):
    documents: List[Document]

class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str] = Field(None)

class ListDatasetsResult(BaseModel):
    datasets: List[Dataset]

class AskParams(BaseModel):
    llmName: str
    datasetNames: List[str]
    llmId: str
    queryString: str
    history: List[Any] = Field(default_factory=list)

class AskResult(BaseModel):
    responseString: str
    citations: List[Dict[str, Any]]

# --- DiscoverTools Models ---
class DiscoverTool(BaseModel):
    dataset_id: str
    dataset_name: str
    description: Optional[str] = None

class ListDiscoverToolsResult(BaseModel):
    tools: List[DiscoverTool]

# === Utility Functions ===
def build_gaia_params(p: ExecuteParams) -> Dict[str, Any]:
    q: Dict[str, Any] = {}
    # Required Gaia parameter
    q["objectTypes"] = p.object_types or ["file"]
    if p.semantic_search_string:
        q["semanticSearchString"] = p.semantic_search_string
    if p.keyword:
        q["keyword"] = p.keyword
    return q

async def call_gaia(params: ExecuteParams) -> List[Dict[str, Any]]:
    url = f"{GAIA_HOST}/v2/mcm/gaia/objects"
    headers = {"accept": "application/json", "apiKey": API_KEY_HEADER}
    logger.debug(f"Gaia objects request URL: {url}")
    logger.debug(f"Gaia request headers: {headers}")
    logger.debug(f"Gaia request params: {build_gaia_params(params)}")
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, params=build_gaia_params(params))
    logger.debug(f"Gaia objects response status: {r.status_code}")
    logger.debug(f"Gaia objects response body: {r.text}")
    r.raise_for_status()
    return r.json().get("objects", [])

async def list_datasets() -> ListDatasetsResult:
    url = f"{GAIA_HOST}/v2/mcm/gaia/datasets"
    headers = {
        "accept": "application/json",
        "apiKey": API_KEY_HEADER
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
    resp.raise_for_status()
    ds = resp.json().get("datasets", [])
    # Map raw list to Dataset models
    items = [{"id": d.get("id"), "name": d.get("name"), "description": d.get("description")} for d in ds]
    return ListDatasetsResult(datasets=items)

async def list_datasets_descriptions() -> ListDiscoverToolsResult:
    """
    Return list of datasets with their descriptions via Gaia discovery API.
    """
    # Fetch all datasets
    ds_url = f"{GAIA_HOST}/v2/mcm/gaia/datasets"
    headers = {
        "accept": "application/json",
        "apiKey": API_KEY_HEADER
    }
    async with httpx.AsyncClient() as client:
        resp_ds = await client.get(ds_url, headers=headers)
    resp_ds.raise_for_status()
    ds_list = resp_ds.json().get("datasets", [])

    # For each dataset, fetch its discovery description
    tasks = []
    async with httpx.AsyncClient() as client:
        for d in ds_list:
            ds_id = d.get("id")
            disc_url = f"{GAIA_HOST}/v2/mcm/gaia/dataset/{ds_id}/discovery?level=1&numLevels=2"
            tasks.append(client.get(disc_url, headers=headers))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    tools: List[DiscoverTool] = []
    for d, r in zip(ds_list, responses):
        desc: Optional[str] = None
        if not isinstance(r, Exception):
            jr = r.json()
            results = jr.get("results", [])
            desc = results[0].get("description", "") if results else None
        tools.append(DiscoverTool(
            dataset_id=d.get("id", ""),
            dataset_name=d.get("name", ""),
            description=desc
        ))

    return ListDiscoverToolsResult(tools=tools)

async def search_objects(
    params: ExecuteParams
) -> ExecuteResult:
    """
    Search via Cohesity Gaia with semantic and facet filters.
    """
    # Call the generic Gaia objects helper
    objects = await call_gaia(params) or []
    # Wrap raw objects into Document models
    docs: List[Document] = []
    for o in objects:
        docs.append(Document(
            id=o.get("id", ""),
            text=o.get("text"),
            metadata=o
        ))
    return ExecuteResult(documents=docs)

async def gaia_qa(params: AskParams):
    url = f"{GAIA_HOST}/v2/mcm/gaia/ask"
    logger.debug(f"Gaia QA request URL: {url}")
    logger.debug(f"Gaia QA headers: {{'accept':'application/json','apiKey': {API_KEY_HEADER}}}")
    logger.debug(f"Gaia QA payload: {params.dict()}")
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "apiKey": API_KEY_HEADER
    }
    payload = params.dict()
    # Use a longer timeout for potentially long-running QA queries
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(url, headers=headers, json=payload, timeout=60.0)
    logger.debug(f"Gaia QA response status: {resp.status_code}")
    logger.debug(f"Gaia QA response body: {resp.text}")
    resp.raise_for_status()
    data = resp.json()
    # Extract the free-form answer string
    resp_str = data.get("responseString", "")
    # Flatten all citations from each document
    citations_list: List[Dict[str, Any]] = []
    docs = data.get("documents") or []
    for doc in docs:
        citations = doc.get("citations") or []
        citations_list.extend(citations)
    return AskResult(responseString=resp_str, citations=citations_list)


# === Discover Tools Endpoint ===
async def discover_tools():
    """
    Discover available datasets and their descriptions via Gaia discovery API.
    """
    # Fetch list of datasets
    ds_url = f"{GAIA_HOST}/v2/mcm/gaia/datasets"
    headers = {"accept": "application/json", "content-type": "application/json", "apiKey": API_KEY_HEADER}
    async with httpx.AsyncClient() as client:
        resp_ds = await client.get(ds_url, headers=headers)
    resp_ds.raise_for_status()
    ds_list = resp_ds.json().get("datasets", [])

    # For each dataset, fetch its discovery description
    tasks = []
    async with httpx.AsyncClient() as client:
        for d in ds_list:
            ds_id = d.get("id")
            disc_url = f"{GAIA_HOST}/v2/mcm/gaia/dataset/{ds_id}/discovery?level=1&numLevels=2"
            tasks.append(client.get(disc_url, headers=headers))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    tools: List[DiscoverTool] = []
    for d, r in zip(ds_list, responses):
        # default to None if no description found
        desc: Optional[str] = None
        if not isinstance(r, Exception):
            jr = r.json()
            results = jr.get("results", [])
            desc = results[0].get("description", "") if results else ""
        tools.append(DiscoverTool(
            dataset_id=d.get("id", ""),
            dataset_name=d.get("name", ""),
            description=desc
        ))

    return ListDiscoverToolsResult(tools=tools)

# # === FastMCP Setup (after route definitions) ===
# mcp = FastMCP.from_fastapi(
#     app,
#     stateless_http=True,
#     http_host="0.0.0.0",
#     http_port=8000
# )

# def start_server():
#     # You can also do: `fastmcp run mcp_service.py` (CLI) or:
#     import uvicorn
#     uvicorn.run("mcp_service:app", host="0.0.0.0", port=8000, reload=True)

# # === Entrypoint ===
# if __name__ == "__main__":
#     start_server()