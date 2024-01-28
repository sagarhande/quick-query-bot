from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from utils import *

router = APIRouter()


# Schemas
class QueryBody(BaseModel):
    user_id: str
    query: str


class KnowledgebaseBody(BaseModel):
    user_id: str
    urls: list


class SummeryBody(BaseModel):
    user_id: str
    url: str


# Helper functions
def validate_input_payload(body: BaseModel):
    return True, body


# Endpoints
@router.post("/ask-question")
async def ask(body: QueryBody):
    is_valid, body = validate_input_payload(body=body)
    if not is_valid:
        return HTTPException(status_code=400, detail=f"Invalid payload")

    return chat_with_llm(query=body.query, user_id=body.user_id)


# Endpoints
@router.post("/extract-summary")
async def extract_summary(body: SummeryBody):
    is_valid, body = validate_input_payload(body=body)
    if not is_valid:
        return HTTPException(status_code=400, detail=f"Invalid payload")

    return summarize_webpage(user_id=body.user_id, url=body.url)


@router.post("/create-knowledgebase")
async def create_knowledgebase(body: KnowledgebaseBody):
    is_valid, body = validate_input_payload(body=body)
    if not is_valid:
        return HTTPException(status_code=400, detail=f"Invalid payload")

    _ = process_urls(urls=body.urls, user_id=body.user_id)
    return {"message": "success"}
