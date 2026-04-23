import json
import re
from typing import List, Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from search import search_recipes

llm = OllamaLLM(model="llama3.1", temperature=0)


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

- "add"    — DEFAULT. Use this for almost everything:
             adding ingredients ("with chickpeas too", "also garlic"),
             removing ingredients ("no chicken", "without beef", "i don't want X"),
             adding constraints ("make it low calorie", "under 500 calories"),
             asking a question about a specific recipe ("what are the ingredients of the first one?",
               "how long does recipe 2 take?", "tell me more about the chicken one"),
             any vague follow-up that builds on current results.

- "more"   — user wants MORE recipes with the EXACT SAME criteria, no changes at all
             (e.g. "show me more", "any other options?", "different ones", "give me more").

- "select" — user is ONLY picking a recipe with NO question attached
             (e.g. "i'll take the first one", "i choose option 2", "i want that one").
             Do NOT use this if the message contains a question.

- "change" — user explicitly wants to START COMPLETELY OVER, discarding all history
             (e.g. "forget it", "start over", "search for something completely different",
              "never mind, show me pizza instead").

IMPORTANT: When in doubt, use "add". Only use "change" if the user clearly wants to
discard everything. Only use "select" if the message is a pure selection with no question.

Return ONLY the single word. No punctuation. No explanation."""),
    ("human", "User message: {message}"),
])

extract_chain = EXTRACT_PROMPT | llm
summary_chain = SUMMARY_PROMPT | llm
intent_chain  = INTENT_PROMPT  | llm

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
    - "title"      : append new keywords (space-separated), no duplicates
    - List fields  : union, no duplicates
    - Numeric scalars: update wins only if not null
    - max_results  : always 3
    """
    merged = base.copy()
    LIST_FIELDS = {"ingredients", "excluded_ingredients", "excluded_title_keywords"}

    for key, new_val in update.items():
        if new_val is None:
            continue

        if key == "title":
            existing = merged.get("title") or ""
            new_title = str(new_val).strip()
            if new_title and new_title.lower() not in existing.lower():
                merged["title"] = f"{existing} {new_title}".strip() if existing else new_title

        elif key in LIST_FIELDS:
            existing = merged.get(key) or []
            merged[key] = existing + [v for v in new_val if v not in existing] or None

        else:
            merged[key] = new_val

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
    for intent in ("select", "more", "change", "add"):
        if intent in raw:
            return intent
    return "add"  # "add" is the safe default


def pick_recipe(message: str, all_recipes: list, latest_recipes: list) -> Optional[dict]:
    """Title-keyword match across all shown, then ordinal match in latest batch."""
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


class SearchSessionState:
    """
    Per-session browsing state.

    Intent routing:
      "add"    -> merge new params onto last_params, re-search  (DEFAULT)
      "more"   -> same params, bigger fetch, exclude seen titles
      "select" -> pick recipe; return it along with the original message
                  so app.py can immediately answer any embedded question
      "change" -> clear all state, fresh search
    """

    def __init__(self):
        self.all_shown_recipes: List[dict] = []
        self.latest_recipes:    List[dict] = []
        self.last_params:       dict       = {}

    def process_message(self, message: str) -> dict:
        """
        Returns:
          {
            "action":        "search"|"add"|"more"|"select"|"change"|"error",
            "message":       <text for chat bubble>,
            "recipe":        <full recipe dict>    # only when action == "select"
            "user_question": <original message>    # only when action == "select"
          }
        """
        if not self.latest_recipes:
            return self._do_search(message)

        try:
            intent = classify_intent(message)
        except Exception:
            intent = "add"

        if intent == "select":
            return self._do_select(message)
        elif intent == "more":
            return self._do_more()
        elif intent == "change":
            return self._do_change(message)
        else:  # "add" — the default
            return self._do_add(message)

    def _do_search(self, user_input: str) -> dict:
        try:
            params = parse_params(user_input)
        except (json.JSONDecodeError, ValueError) as e:
            return {"action": "error", "message": f"Sorry, I couldn't understand that. ({e})"}

        recipes = search(params)
        if not recipes:
            return {"action": "error", "message": "I couldn't find any recipes matching that. Try a different search!"}

        self.last_params    = params
        self.latest_recipes = recipes
        self.all_shown_recipes.extend(recipes)
        return {"action": "search", "message": summarise(recipes)}

    def _do_add(self, user_input: str) -> dict:
        """Extract params from new message, merge with existing, re-search."""
        try:
            new_params = parse_params(user_input)
        except (json.JSONDecodeError, ValueError) as e:
            return {"action": "error", "message": f"Sorry, I couldn't understand that. ({e})"}

        merged  = merge_params(self.last_params, new_params)
        recipes = search(merged)

        if not recipes:
            return {"action": "error", "message": "I couldn't find recipes matching those combined criteria. Try relaxing some constraints!"}

        self.last_params    = merged
        self.latest_recipes = recipes
        existing_titles = {r.get("title") for r in self.all_shown_recipes}
        for r in recipes:
            if r.get("title") not in existing_titles:
                self.all_shown_recipes.append(r)

        return {"action": "add", "message": summarise(recipes)}

    def _do_more(self) -> dict:
        if not self.last_params:
            return {"action": "error", "message": "I don't have a previous search to extend. What are you looking for?"}

        more_results = search({**self.last_params, "max_results": 15})
        shown_titles = {r.get("title") for r in self.all_shown_recipes}
        new_recipes  = [r for r in more_results if r.get("title") not in shown_titles][:3]

        if not new_recipes:
            return {"action": "error", "message": "Sorry, no more recipes matching your criteria. Try changing your search!"}

        self.latest_recipes = new_recipes
        for r in new_recipes:
            self.all_shown_recipes.append(r)
        return {"action": "more", "message": summarise(new_recipes)}

    def _do_change(self, user_input: str) -> dict:
        self.all_shown_recipes = []
        self.latest_recipes    = []
        self.last_params       = {}
        return self._do_search(user_input)

    def _do_select(self, message: str) -> dict:
        selected = pick_recipe(message, self.all_shown_recipes, self.latest_recipes)
        if not selected:
            return {
                "action":  "error",
                "message": "I couldn't tell which recipe you meant. "
                           "Try 'the first one', 'the second one', or say the recipe name.",
            }
        return {
            "action":        "select",
            "message":       f"You've selected **{selected.get('title')}**.",
            "recipe":        selected,
            "user_question": message,  
        }


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
                print(f"\nASSISTANT: {active_assistant.ask(user_input, cli_session_id)}\n")
                continue

            result = session.process_message(user_input)

            if result["action"] == "select":
                recipe_text      = json.dumps(result["recipe"], indent=2, ensure_ascii=False)
                active_assistant = RecipeAssistant(recipe_text)
                print(f"\nASSISTANT: {active_assistant.ask(result['user_question'], cli_session_id)}\n")
            else:
                print(f"\nASSISTANT: {result['message']}\n")

        except KeyboardInterrupt:
            print("\nExiting agent...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")