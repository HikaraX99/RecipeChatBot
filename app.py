import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from llm_pre_es import run_agent_full, classify_intent, search, summarise
from llm_handler import RecipeAssistant

app = FastAPI(title="Recipe Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

assistants: dict[str, RecipeAssistant] = {}
search_sessions: dict[str, dict] = {}

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

def pick_recipe(message: str, recipes: list) -> Optional[dict]:
    msg = message.lower()
    ordinals = {
        "first": 0, "1st": 0, "one": 0, "1": 0,
        "second": 1, "2nd": 1, "two": 1, "2": 1,
        "third": 2, "3rd": 2, "three": 2, "3": 2,
    }
    for word, idx in ordinals.items():
        if word in msg and idx < len(recipes):
            return recipes[idx]
    # Title keyword match
    for recipe in recipes:
        title_words = recipe.get("title", "").lower().split()
        if any(w in msg for w in title_words if len(w) > 3):
            return recipe
    return recipes[0] if recipes else None


@app.post("/search", response_model=SearchResponse)
def search_recipes_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    result = run_agent_full(req.query)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    session_id = str(uuid.uuid4())
    search_sessions[session_id] = result  # store for follow-up routing

    return SearchResponse(summary=result["summary"], session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Handles all follow-up messages after /search.

    If the session still has pending search state (user hasn't selected a recipe yet),
    classify the user's intent:
      - 'select' → pick a recipe, init the assistant, reply with confirmation
      - 'more'   → search again with same params, return new summary
      - 'change' → treat message as new search query, return new summary

    Once a recipe is selected (assistant exists), route directly to the assistant
    as before — no change in behaviour.
    """

    if req.session_id in assistants:
        assistant = assistants[req.session_id]
        answer = assistant.ask(req.message, req.session_id)
        return ChatResponse(answer=answer)

    if req.session_id in search_sessions:
        session = search_sessions[req.session_id]
        intent = classify_intent(req.message)

        if intent == "select":
            recipe = pick_recipe(req.message, session["recipes"])
            if not recipe:
                return ChatResponse(answer="I couldn't tell which recipe you meant. "
                                           "Try saying 'the first one' or the recipe name.")
            # Init assistant with the selected recipe
            recipe_text = json.dumps(recipe, indent=2, ensure_ascii=False)
            assistants[req.session_id] = RecipeAssistant(recipe_text)
            # Clean up search state — not needed anymore
            del search_sessions[req.session_id]
            return ChatResponse(
                answer=f"Great choice! You've selected **{recipe.get('title')}**. "
                       "Ask me anything about it — ingredients, steps, substitutions, nutrition, etc."
            )

        elif intent == "more":
            params = {**session["params"], "max_results": 6}
            all_recipes = search(params)
            shown = {r.get("title") for r in session["recipes"]}
            new_recipes = [r for r in all_recipes if r.get("title") not in shown][:3]

            if not new_recipes:
                return ChatResponse(answer="Sorry, I couldn't find more recipes matching "
                                           "your criteria. Try changing your requirements!")

            new_summary = summarise(new_recipes)
            search_sessions[req.session_id]["recipes"] = new_recipes
            search_sessions[req.session_id]["summary"] = new_summary
            return ChatResponse(answer=new_summary)

        else:  # intent == "change"
            result = run_agent_full(req.message)
            if "error" in result:
                return ChatResponse(answer=result["error"])
            search_sessions[req.session_id] = result
            return ChatResponse(answer=result["summary"])

    if not req.recipe_text:
        raise HTTPException(
            status_code=400,
            detail="recipe_text is required for the first message in a session."
        )
    assistants[req.session_id] = RecipeAssistant(req.recipe_text)
    assistant = assistants[req.session_id]
    answer = assistant.ask(req.message, req.session_id)
    return ChatResponse(answer=answer)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Optional — clear a session to free memory."""
    assistants.pop(session_id, None)
    search_sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/health")
def health():
    return {"status": "ok"}