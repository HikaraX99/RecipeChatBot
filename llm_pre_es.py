import json
import re
from typing import List, Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from search import search_recipes

llm = OllamaLLM(model="llama3.1", temperature=0)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract recipe search parameters from the user message and return ONLY a JSON object.

Rules:
- "title": ingredient or dish name to search for (e.g. "chicken", "pasta"). Use null if none mentioned.
- "max_calories": number if mentioned or implied ("light"->500, "big meal"->1000), else null
- "min_calories": number if mentioned, else null
- "min_protein": number if implied ("high protein"->25, "post workout"->30, "strength training"->30), else null
- "max_protein": number if mentioned, else null
- "max_fat": number if implied ("low fat"->10), else null
- "min_fat": number if mentioned, else null
- "max_carbs": number if implied ("low carb"->20, "keto"->20), else null
- "min_carbs": number if mentioned, else null
- "min_fiber": number if implied ("high fiber"->8), else null
- "max_fiber": number if mentioned, else null
- "min_sugar": number if mentioned, else null
- "max_sugar": number if mentioned, else null
- "min_sodium": number if mentioned, else null
- "max_sodium": number if mentioned, else null
- "ingredients": list of ingredients the user WANTS in the recipe, else null
- "excluded_ingredients": list of ingredients the user explicitly does NOT want (e.g. "no chicken", "without garlic", "i don't want beef"), else null
- "excluded_title_keywords": list of words that should NOT appear in the recipe title, else null
- "max_results": always 3

Return ONLY the JSON object. No explanation. No markdown. No extra text."""),
    ("human", "{user_input}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful cooking assistant.
Summarise the recipes below. For each one show:
- Title
- Calories and protein
- One sentence description

Use ONLY the recipes provided. Do not invent anything. Do not ask questions."""),
    ("human", "Recipes:\n{recipes}"),
])

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's intent after they've seen recipe suggestions.
Return ONLY one of these exact strings:

- "add"    — user wants to ADD a new ingredient or constraint ON TOP of the current search
             (e.g. "with chickpeas too", "also add garlic", "i want chickpeas as well",
              "and make it spicy", "no chicken", "without beef", "i don't want X")

- "more"   — user wants MORE recipes with the SAME criteria, no new constraints
             (e.g. "show me more", "any other options?", "different ones", "give me more")

- "select" — user is PICKING one of the shown recipes
             (e.g. "i'll take the first one", "option 2", "the chicken one",
              "tell me more about the second", "i want that one")

- "change" — user wants to START OVER with COMPLETELY NEW criteria, replacing everything
             (e.g. "actually find me pasta", "forget it, show me salads",
              "i want something totally different", "search for pizza instead")

When in doubt between "add" and "change": if the message mentions keeping or extending
current results, use "add". If it sounds like a fresh start, use "change".

Return ONLY the single word. No punctuation. No explanation."""),
    ("human", "User message: {message}"),
])

extract_chain = EXTRACT_PROMPT | llm
summary_chain = SUMMARY_PROMPT | llm
intent_chain  = INTENT_PROMPT  | llm

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def parse_params(user_input: str) -> dict:
    raw = extract_chain.invoke({"user_input": user_input})
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)
    return json.loads(clean)


def merge_params(base: dict, update: dict) -> dict:
    """
    Merge new extracted params ON TOP of existing params so constraints accumulate.

    - "title": APPEND new title keywords to existing ones (space-separated)
               so "chicken" + "chickpeas" -> "chicken chickpeas" in the ES query.
    - List fields (ingredients, excluded_ingredients, excluded_title_keywords):
               union — no duplicates.
    - Numeric scalars: update wins only if not null (keeps existing otherwise).
    - max_results: always 3.
    """
    merged = base.copy()

    LIST_FIELDS = {"ingredients", "excluded_ingredients", "excluded_title_keywords"}

    for key, new_val in update.items():
        if new_val is None:
            continue  # keep existing value when update has nothing new

        if key == "title":
            existing_title = merged.get("title") or ""
            new_title = str(new_val).strip()
            # Append only if the new keyword isn't already in the existing title
            if new_title and new_title.lower() not in existing_title.lower():
                merged["title"] = f"{existing_title} {new_title}".strip() if existing_title else new_title
            # If new_title is already covered, leave existing unchanged

        elif key in LIST_FIELDS:
            existing = merged.get(key) or []
            combined = existing + [v for v in new_val if v not in existing]
            merged[key] = combined if combined else None

        else:
            merged[key] = new_val  # numeric scalar: update wins

    merged["max_results"] = 3
    return merged


def search(params: dict) -> List[dict]:
    results = search_recipes.invoke({k: v for k, v in params.items() if v is not None})
    if not results:
        return []
    seen, unique = set(), []
    for r in results:
        t = r.get("title", "")
        if t and t not in seen:
            seen.add(t)
            unique.append(r)
    return unique


def summarise(recipes: List[dict]) -> str:
    slim = [
        {
            "title":       r.get("title"),
            "calories":    r.get("nutrition", {}).get("calories"),
            "protein_g":   r.get("nutrition", {}).get("protein_g"),
            "ingredients": r.get("ingredients", [])[:5],
        }
        for r in recipes
    ]
    return summary_chain.invoke({"recipes": json.dumps(slim, indent=2)})


def classify_intent(message: str) -> str:
    raw = intent_chain.invoke({"message": message}).strip().lower()
    for intent in ("add", "select", "more", "change"):
        if intent in raw:
            return intent
    return "change"  # safe default


def pick_recipe(message: str, all_recipes: list, latest_recipes: list) -> Optional[dict]:
    """
    1. Title-keyword match across ALL ever-shown recipes.
    2. Ordinal match within the latest batch.
    """
    msg = message.lower()

    for recipe in all_recipes:
        title_words = [w for w in recipe.get("title", "").lower().split() if len(w) > 3]
        if title_words and any(w in msg for w in title_words):
            return recipe

    ordinals = {
        "first": 0, "1st": 0, "1": 0,
        "second": 1, "2nd": 1, "2": 1,
        "third": 2, "3rd": 2, "3": 2,
    }
    for word, idx in ordinals.items():
        if word in msg and idx < len(latest_recipes):
            return latest_recipes[idx]

    return None


def run_agent_full(user_input: str) -> dict:
    try:
        params = parse_params(user_input)
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": f"Sorry, I couldn't understand that request. ({e})"}

    recipes = search(params)
    if not recipes:
        return {"error": "I couldn't find any recipes matching that in the database."}

    return {"params": params, "recipes": recipes, "summary": summarise(recipes)}


def run_agent(user_input: str) -> str:
    result = run_agent_full(user_input)
    return result.get("error") or result.get("summary", "")


# ---------------------------------------------------------------------------
# SearchSessionState
# ---------------------------------------------------------------------------

class SearchSessionState:
    """
    Per-session browsing state.

    Intent routing:
      "add"    -> merge new params onto last_params, re-search
      "more"   -> same params, bigger fetch, exclude seen titles
      "select" -> pick recipe from history
      "change" -> clear all state, start fresh search
    """

    def __init__(self):
        self.all_shown_recipes: List[dict] = []
        self.latest_recipes:    List[dict] = []
        self.last_params:       dict       = {}

    def process_message(self, message: str) -> dict:
        """
        Returns:
          { "action": "search"|"add"|"more"|"select"|"change"|"error",
            "message": <str for frontend>,
            "recipe":  <dict>  # only when action == "select" }
        """
        # First message always goes straight to search
        if not self.latest_recipes:
            return self._do_search(message)

        try:
            intent = classify_intent(message)
        except Exception:
            intent = "change"

        if intent == "select":
            return self._do_select(message)
        elif intent == "more":
            return self._do_more()
        elif intent == "add":
            return self._do_add(message)
        else:  # "change"
            return self._do_change(message)

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _do_search(self, user_input: str) -> dict:
        """Initial search — parses params and runs ES query."""
        try:
            params = parse_params(user_input)
        except (json.JSONDecodeError, ValueError) as e:
            return {"action": "error",
                    "message": f"Sorry, I couldn't understand that request. ({e})"}

        recipes = search(params)
        if not recipes:
            return {"action": "error",
                    "message": "I couldn't find any recipes matching that. Try a different search!"}

        self.last_params    = params
        self.latest_recipes = recipes
        self.all_shown_recipes.extend(recipes)

        return {"action": "search", "message": summarise(recipes)}

    def _do_add(self, user_input: str) -> dict:
        """
        ADD intent: extract params from the new message, MERGE with last_params,
        re-search. Title keywords are appended, list fields are unioned.
        e.g. last_params had title="chicken"; user says "with chickpeas as well"
             -> new params title="chickpeas", ingredients=["chickpeas"]
             -> merged title="chicken chickpeas", ingredients=["chickpeas"]
        """
        try:
            new_params = parse_params(user_input)
        except (json.JSONDecodeError, ValueError) as e:
            return {"action": "error",
                    "message": f"Sorry, I couldn't understand that. ({e})"}

        merged = merge_params(self.last_params, new_params)
        recipes = search(merged)

        if not recipes:
            return {"action": "error",
                    "message": "I couldn't find recipes matching those combined criteria. "
                               "Try relaxing some constraints!"}

        self.last_params    = merged
        self.latest_recipes = recipes

        existing_titles = {r.get("title") for r in self.all_shown_recipes}
        for r in recipes:
            if r.get("title") not in existing_titles:
                self.all_shown_recipes.append(r)

        return {"action": "add", "message": summarise(recipes)}

    def _do_more(self) -> dict:
        """MORE intent: same params, bigger fetch, exclude already-seen titles."""
        if not self.last_params:
            return {"action": "error",
                    "message": "I don't have a previous search to extend. What are you looking for?"}

        more_results = search({**self.last_params, "max_results": 15})
        shown_titles = {r.get("title") for r in self.all_shown_recipes}
        new_recipes  = [r for r in more_results if r.get("title") not in shown_titles][:3]

        if not new_recipes:
            return {"action": "error",
                    "message": "Sorry, no more recipes matching your criteria. Try changing your search!"}

        self.latest_recipes = new_recipes
        for r in new_recipes:
            self.all_shown_recipes.append(r)

        return {"action": "more", "message": summarise(new_recipes)}

    def _do_change(self, user_input: str) -> dict:
        """CHANGE intent: wipe all state and start a completely fresh search."""
        self.all_shown_recipes = []
        self.latest_recipes    = []
        self.last_params       = {}
        return self._do_search(user_input)

    def _do_select(self, message: str) -> dict:
        """SELECT intent: match message to a recipe; ask to clarify if ambiguous."""
        selected = pick_recipe(message, self.all_shown_recipes, self.latest_recipes)

        if not selected:
            return {
                "action":  "error",
                "message": "I couldn't tell which recipe you meant. "
                           "Try 'the first one', 'the second one', or say the recipe name.",
            }

        return {
            "action":  "select",
            "message": f"Great choice! You've selected **{selected.get('title')}**. "
                       "Ask me anything about it — ingredients, steps, substitutions, nutrition, etc.",
            "recipe":  selected,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from llm_handler import RecipeAssistant

    print("Recipe Agent started. Type 'quit' or 'exit' to stop.\n")

    session          = SearchSessionState()
    active_assistant = None
    cli_session_id   = "cli_user_1"

    while True:
        try:
            user_input = input("USER: ").strip()
            if user_input.lower() in ("quit", "exit"):
                print("Exiting agent...")
                break
            if not user_input:
                continue

            if active_assistant:
                answer = active_assistant.ask(user_input, cli_session_id)
                print(f"\nASSISTANT: {answer}\n")
                continue

            result = session.process_message(user_input)
            print(f"\nASSISTANT: {result['message']}\n")

            if result["action"] == "select":
                recipe_text      = json.dumps(result["recipe"], indent=2, ensure_ascii=False)
                active_assistant = RecipeAssistant(recipe_text)

        except KeyboardInterrupt:
            print("\nExiting agent...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")