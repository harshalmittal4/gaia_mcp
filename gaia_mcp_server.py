import json
from typing import Any, Dict, List, Optional, NamedTuple, Tuple
from fastapi import FastAPI, HTTPException
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ClientError
from fastmcp import FastMCP
from gaia_service import *

# === FastAPI App & FastMCP Setup ===
app = FastAPI()

# # === The MCP Server ===
# class GaiaMCPClient:
#     """
#     A wrapper around a FastMCP-powered Gaia MCP Service.

#     Usage:
#         async with GaiaMCPClient(mcp_url, api_key) as client:
#             datasets = await client.list_datasets()
#             tools = await client.discover_tools()
#             answer, citations = await client.ask([datasets[0].name], "Your question?")
#             tool_descriptions = await client.list_tool_descriptions()
#     """

#     def __init__(
#         self,
#         mcp_url: str,
#         api_key: str
#     ):
#         transport = StreamableHttpTransport(
#             url=mcp_url,
#             headers={"X-API-Key": api_key}
#         )
#         self._client = Client(transport=transport)

#     async def __aenter__(self):
#         await self._client.__aenter__()
#         return self

#     async def __aexit__(
#         self, exc_type, exc_val, exc_tb
#     ):
#         await self._client.__aexit__(exc_type, exc_val, exc_tb)

# @app.post(
#     "/list_datasets",
#     response_model=List[Dataset],
#     operation_id="list_datasets",
#     summary="List all accessible datasets in Gaia",
#     description="""
# Tool Name: List Datasets

# Purpose:
#     Retrieves a list of all datasets that are accessible to the authenticated user within the Cohesity Gaia instance.

# Inputs:
#     This endpoint does not require any input parameters.

# Output:
#     A list of dataset objects. Each object includes:
#     - id (str): A unique identifier for the dataset.
#     - name (str): The human-readable name of the dataset.

# Usage Notes:
#     - If no datasets are accessible, an empty list will be returned.
#     - This tool is useful for discovering dataset names and IDs for use with other tools (e.g., querying datasets via Gaia QA).
# """
# )
# async def list_datasets_endpoint() -> List[dict]:
#     try:
#         raw = await list_datasets()
#         if not raw:
#             return []
#         return [
#             {
#                 "id": d.id,
#                 "name": d.name
#             }
#             for d in raw.datasets
#         ]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in list_datasets: {str(e)}")

# @app.post(
#     "/list_datasets_descriptions",
#     response_model=List[DiscoverTool],
#     operation_id="list_datasets_descriptions",
#     summary="List datasets with descriptions and content summaries",
#     description="""
# Tool Name: List Dataset Descriptions

# Purpose:
#     Retrieves a list of all available datasets along with high-level descriptions summarizing the contents,
#     topics, and themes discussed in the documents within each dataset. Useful for understanding dataset
#     relevance before querying.

# Inputs:
#     This endpoint does not require any input parameters.

# Output:
#     A list of dataset description objects. Each object includes:
#     - dataset_id (str): A unique identifier for the dataset.
#     - dataset_name (str): The human-readable name of the dataset.
#     - description (str, optional): A brief summary of the dataset's content, including dominant themes or topics.

# Usage Notes:
#     - Use this tool to explore available datasets before formulating search or QA queries.
#     - Datasets with no description will return `null` or empty string in the `description` field.
#     - Ideal for dataset discovery workflows in automated agents or user interfaces.
# """
# )
# async def list_datasets_descriptions_endpoint() -> List[dict]:
#     try:
#         raw = await list_datasets_descriptions()
#         if not raw:
#             return []
#         return [
#             {
#                 "dataset_id": t.dataset_id,
#                 "dataset_name": t.dataset_name,
#                 "description": t.description
#             }
#             for t in raw.tools
#         ]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in list_datasets_descriptions: {str(e)}")

# @app.post(
#     "/search_objects",
#     response_model=List[Document],
#     operation_id="search_objects",
#     summary="Search and filter content objects from Gaia",
#     description="""
# Tool Name: Search Objects

# Purpose:
#     This tool retrieves a list of available objects from Gaia based on filtering criteria such as keyword,
#     semantic search string, object types, file types, and file size range. It supports both keyword-based
#     and semantic (vector-based) search.

# Inputs:
#     - keyword (str, optional): A keyword to match in object content or metadata.
#     - semantic_search_string (str, optional): A natural language query string used for semantic (vector) search.
#     - object_types (List[str], optional): List of object types to include in the result (e.g., "pdf", "email").
#     - file_type (List[str], optional): List of file types to include (e.g., "docx", "txt").
#     - file_gt_kb (int, optional): Return only objects with file size greater than this value in kilobytes.
#     - file_lt_kb (int, optional): Return only objects with file size less than this value in kilobytes.

# Output:
#     A list of Document objects. Each object includes:
#     - id (str): A unique identifier for the object.
#     - text (str, optional): Extracted text content from the object, if available.
#     - metadata (dict): Metadata associated with the object (e.g., filename, size, type, date created).

# Usage Notes:
#     - You can combine multiple filters (e.g., keyword + file type + size range).
#     - If both `keyword` and `semantic_search_string` are provided, both will be used in conjunction.
#     - Returns an empty list if no matching objects are found.
# """
# )
# async def search_objects_endpoint(
#     keyword: Optional[str] = None,
#     semantic_search_string: Optional[str] = None,
#     object_types: Optional[List[str]] = None,
#     file_type: Optional[List[str]] = None,
#     file_gt_kb: Optional[int] = None,
#     file_lt_kb: Optional[int] = None
# ) -> List[dict]:
#     try:
#         params = ExecuteParams(
#             keyword=keyword,
#             semantic_search_string=semantic_search_string,
#             object_types=object_types,
#             file_type=file_type,
#             file_greater_than_kb=file_gt_kb,
#             file_less_than_kb=file_lt_kb
#         )

#         raw = await search_objects(params)
#         return [
#             {
#                 "id": d.id,
#                 "text": d.text,
#                 "metadata": d.metadata
#             }
#             for d in raw.documents
#         ]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in search_objects: {str(e)}")
    
@app.post(
    "/gaia_qa",
    response_model=dict,
    operation_id="ask",
    summary="Query datasets using a specified LLM and return answers with citations.",
    description="""
Tool Name: Gaia QA

Purpose:
    This tool allows querying one or more datasets using a Large Language Model (LLM). 
    It processes a user-specified question, optionally with conversation history, and returns a response 
    generated by the LLM along with the list of citations (sources) used in producing the answer.

Inputs:
    - question (str, required): The natural language question to be answered by the LLM.
    - dataset_names (List[str], optional): A list of dataset names to search the query against. Default is ["data_explorer_create_themes_test"].
    - llm_name (str, optional): The name of the LLM to use. Default is "Cohesity LLM Advanced".
    - llm_id (str, optional): The identifier for the LLM to be used. Default is "ADV".
    - history (List[Any], optional): List of prior interactions or query history, used to provide context.

Output:
    A dictionary with the following keys:
    - responseString (str): The answer string generated by the LLM.
    - citations (List[dict]): A list of citation objects (e.g., documents, source snippets) 
      that were referenced to generate the answer.

Usage Notes:
    - If no LLM is specified, defaults are used.
    - This endpoint is asynchronous and supports chat-style context via the `history` parameter.
"""
)
async def ask_endpoint(
    question: str,
    dataset_names: List[str] = ["ashok_test", "vpangha_qure6"], #data_explorer_create_themes_test
    llm_name: str = "Cohesity LLM Advanced",
    llm_id:   str = "ADV",
    history:  List[Any] = []
) -> dict:
    try:
        params = AskParams(
            llmName=llm_name,
            datasetNames=dataset_names,
            llmId=llm_id,
            queryString=question,
            history=history
        )
        raw = await gaia_qa(params)
        return {
            "responseString": raw.responseString,
            "citations": raw.citations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ask: {str(e)}")

# @app.post(
#     "/discover_tools",
#     response_model=List[DiscoverTool],
#     operation_id="discover_tools",
#     summary="Discover tools",
#     description="""
# Description:
#     Discovers available datasets and their descriptions via the Gaia discovery API.

# Input:
#     This endpoint does not require any input parameters.

# Output:
#     A list of DiscoverTool objects, each containing:
#         - dataset_id: The unique identifier of the dataset (str)
#         - dataset_name: The name of the dataset (str)
#         - description: A description of the dataset's contents (str, optional)
# """
# )
# async def discover_tools_endpoint() -> List[dict]:
#     try:
#         raw = await discover_tools()
#         if not raw:
#             return []
#         return [
#             {
#                 "dataset_id": t.dataset_id,
#                 "dataset_name": t.dataset_name,
#                 "description": t.description
#             }
#             for t in raw.tools
#         ]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in discover_tools: {str(e)}")

    # async def list_tool_descriptions(self) -> Dict[str, Optional[str]]:
    #     """
    #     Call the 'list_tool_descriptions' tool on the MCP server
    #     and return a mapping of tool names to their descriptions.
    #     """
    #     raw = await self._client.call_tool("list_tool_descriptions", {})
    #     if not raw:
    #         return {}
    #     first = raw[0]
    #     payload = json.loads(first.text) if hasattr(first, "text") else first
    #     # Assume server returns {"tools": [{"name": "...", "description": "..."}, ...]}
    #     return {
    #         t.get("name", ""): t.get("description")
    #         for t in payload.get("tools", [])
    #     }

# === Health Check ===
@app.get(
    "/healthz",
    summary="Health check",
    description="""
Description:
    Health check endpoint to verify the MCP server is running.

Input:
    This endpoint does not require any input parameters.

Output:
    A JSON object with a status key indicating server health.
"""
)
def health_endpoint() -> dict:
    try:
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in health: {str(e)}")

# Register with FastMCP
mcp = FastMCP.from_fastapi(
    app,
    stateless_http=True,
    http_host="0.0.0.0",
    http_port=8001  # Use a different port if needed
)

if __name__ == "__main__":
    mcp.run()