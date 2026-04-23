import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from llm_pre_es import SearchSessionState
from llm_handler import RecipeAssistant

app = FastAPI(title="Recipe Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

assistants:      dict[str, RecipeAssistant]    = {}
search_sessions: dict[str, SearchSessionState] = {}


class SearchRequest(BaseModel):
    query: str

class RecipeCard(BaseModel):
    id:        str   
    title:     str
    image_url: str   

class SearchResponse(BaseModel):
    summary:      str
    session_id:   str
    recipe_cards: List[RecipeCard] = []  

class ChatRequest(BaseModel):
    session_id:  str
    message:     str
    recipe_text: Optional[str] = None

class ChatResponse(BaseModel):
    answer:       str
    recipe_cards: List[RecipeCard] = []
    recipe_text:  Optional[str] = None


def to_card(recipe: dict) -> RecipeCard:
    return RecipeCard(
        id        = recipe.get("source_url", ""),
        title     = recipe.get("title", ""),
        image_url = recipe.get("image", ""),
    )


@app.post("/search", response_model=SearchResponse)
def search_recipes_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    session    = SearchSessionState()
    result     = session.process_message(req.query)
    session_id = str(uuid.uuid4())

    if result["action"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    search_sessions[session_id] = session
    return SearchResponse(
        summary      = result["message"],
        session_id   = session_id,
        recipe_cards = [to_card(r) for r in session.latest_recipes],
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if req.session_id in assistants:
        answer = assistants[req.session_id].ask(req.message, req.session_id)
        return ChatResponse(answer=answer)

    if req.session_id in search_sessions:
        session = search_sessions[req.session_id]
        result  = session.process_message(req.message)

        if result["action"] == "select":
            recipe_text = json.dumps(result["recipe"], indent=2, ensure_ascii=False)
            assistant   = RecipeAssistant(recipe_text)
            assistants[req.session_id] = assistant
            del search_sessions[req.session_id]

            answer = assistant.ask(result["user_question"], req.session_id)
            return ChatResponse(answer=answer, recipe_text=recipe_text)

        cards = []
        if result["action"] in ("search", "add", "more", "change"):
            cards = [to_card(r) for r in session.latest_recipes]

        return ChatResponse(answer=result["message"], recipe_cards=cards)

    if not req.recipe_text:
        raise HTTPException(
            status_code=400,
            detail="No active session found. Start a new search via /search, "
                   "or supply recipe_text to bootstrap a session directly.",
        )
    assistants[req.session_id] = RecipeAssistant(req.recipe_text)
    answer = assistants[req.session_id].ask(req.message, req.session_id)
    return ChatResponse(answer=answer)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    assistants.pop(session_id, None)
    search_sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/health")
def health():
    return {"status": "ok"}
