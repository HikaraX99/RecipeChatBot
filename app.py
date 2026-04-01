import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from llm_pre_es import SearchSessionState
from llm_handler import RecipeAssistant

app = FastAPI(title="Recipe Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keyed by session_id
assistants:      dict[str, RecipeAssistant]    = {}
search_sessions: dict[str, SearchSessionState] = {}


# ---------------------------------------------------------------------------
# Models  (shapes unchanged — frontend untouched)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    summary: str
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    recipe_text: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
def search_recipes_endpoint(req: SearchRequest):
    """
    Step 1 — Initial recipe search.
    Creates a SearchSessionState, runs the first search, stores the session.
    Response shape is identical to before: { summary, session_id }.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    session    = SearchSessionState()
    result     = session.process_message(req.query)
    session_id = str(uuid.uuid4())

    if result["action"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    search_sessions[session_id] = session
    return SearchResponse(summary=result["message"], session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Step 2 — All follow-up messages from the frontend.

    Phase 2  (recipe already selected):
      Route directly to RecipeAssistant — no change from original behaviour.

    Phase 1  (user still browsing):
      Delegate entirely to SearchSessionState.process_message(), which mirrors
      the __main__ CLI logic:
        'search' / 'change' → new ES query, return summary
        'more'              → extend previous search, return new summary
        'select'            → pick recipe, init assistant, return confirmation
        'error'             → return error message gracefully (no 500)

    Fallback (direct API call with recipe_text):
      Backwards-compatible with any client that bootstraps a session manually.
    """

    # --- Phase 2: chat with the selected recipe ---
    if req.session_id in assistants:
        answer = assistants[req.session_id].ask(req.message, req.session_id)
        return ChatResponse(answer=answer)

    # --- Phase 1: user is still browsing ---
    if req.session_id in search_sessions:
        session = search_sessions[req.session_id]
        result  = session.process_message(req.message)

        if result["action"] == "select":
            recipe_text = json.dumps(result["recipe"], indent=2, ensure_ascii=False)
            assistants[req.session_id] = RecipeAssistant(recipe_text)
            del search_sessions[req.session_id]   # no longer needed

        return ChatResponse(answer=result["message"])

    # --- Fallback: no session found, try recipe_text bootstrap ---
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
    """Optional — free memory for a session."""
    assistants.pop(session_id, None)
    search_sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/health")
def health():
    return {"status": "ok"}