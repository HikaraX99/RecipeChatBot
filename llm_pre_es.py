import json
import re
from typing import List, Optional
from llm_handler import RecipeAssistant

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from search import search_recipes

llm = OllamaLLM(model="llama3.1", temperature=0)

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract recipe search parameters from the user message and return ONLY a JSON object.

Rules:
- "title": ingredient or dish name only (no diet words like healthy/vegan/quick)
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
- "ingredients": list of ingredient strings the user has available, else null
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
- "select" — user is picking one of the shown recipes (e.g. "I'll take the first one", "option 2", "the chicken one", "tell me more about the second")
- "more"   — user wants more/different recipes with similar criteria (e.g. "show me more", "any other options?", "different ones")
- "change" — user wants to search with new/different criteria (e.g. "actually I want something vegan", "find me pasta instead", "something with less calories")

Return ONLY the single word. No punctuation. No explanation."""),
    ("human", "User message: {message}"),
])

extract_chain = EXTRACT_PROMPT | llm
summary_chain = SUMMARY_PROMPT | llm
intent_chain  = INTENT_PROMPT  | llm  # NEW


def parse_params(user_input: str) -> dict:
    raw = extract_chain.invoke({"user_input": user_input})
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)
    return json.loads(clean)


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
            "title": r.get("title"),
            "calories": r.get("nutrition", {}).get("calories"),
            "protein_g": r.get("nutrition", {}).get("protein_g"),
            "ingredients": r.get("ingredients", [])[:5],
        }
        for r in recipes
    ]
    return summary_chain.invoke({"recipes": json.dumps(slim, indent=2)})


def classify_intent(message: str) -> str:
    """
    Returns 'select', 'more', or 'change'.
    Falls back to 'change' if the model returns something unexpected.
    """
    raw = intent_chain.invoke({"message": message}).strip().lower()
    for intent in ("select", "more", "change"):
        if intent in raw:
            return intent
    return "change"  # safe default


def run_agent_full(user_input: str) -> dict:
    """
    Same as run_agent but returns a dict with params, recipes, and summary
    so app.py can store search state for follow-up handling.
    """
    try:
        params = parse_params(user_input)
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": f"Sorry, I couldn't understand that request. ({e})"}

    recipes = search(params)
    if not recipes:
        return {"error": "I couldn't find any recipes matching that in the database."}

    summary = summarise(recipes)
    return {
        "params": params,
        "recipes": recipes,
        "summary": summary,
    }


def run_agent(user_input: str) -> str:
    try:
        params = parse_params(user_input)
    except (json.JSONDecodeError, ValueError) as e:
        return f"Sorry, I couldn't understand that request. ({e})"

    '''
    print("\n--- RECIPE SEARCH TRIGGERED ---")
    print(f"  Title          : {params.get('title')}")
    print(f"  Cal range      : {params.get('min_calories')} - {params.get('max_calories')} kcal")
    print(f"  Protein range  : {params.get('min_protein')} - {params.get('max_protein')} g")
    print(f"  Fat range      : {params.get('min_fat')} - {params.get('max_fat')} g")
    print(f"  Carbs range    : {params.get('min_carbs')} - {params.get('max_carbs')} g")
    print(f"  Fiber range    : {params.get('min_fiber')} - {params.get('max_fiber')} g")
    print(f"  Ingredients    : {params.get('ingredients')}")
    print(f"  Max results    : {params.get('max_results', 3)}")
    '''

    recipes = search(params)
    if not recipes:
        return "I couldn't find any recipes matching that in the database."

    return summarise(recipes)

# bla
def pick_recipe(message: str, all_recipes: list, latest_recipes: list) -> dict | None:
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


if __name__ == "__main__":
    print("Recipe Agent started. Type 'quit' or 'exit' to stop.")
    
    all_shown_recipes = []   # Remembers EVERY recipe shown so far
    latest_recipes = []      # Only the recipes from the most recent search
    search_history = []      # Remembers previous search conditions
    active_assistant = None
    session_id = "cli_user_1"
    
    while True:
        try:
            user_input = input("\nUSER: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting agent...")
                break
                
            if active_assistant:
                answer = active_assistant.ask(user_input, session_id)
                print(f"\nASSISTANT: {answer}")
                continue
                
            intent = "change"
            if latest_recipes:
                try:
                    intent = classify_intent(user_input)
                except Exception:
                    intent = "change"
            
            if intent == "select":
                while True:
                    try:
                        user_input = input("\nUSER: ").strip()
                        if user_input.lower() in ['quit', 'exit']:
                            print("Exiting agent...")
                            break
                            
                        if active_assistant:
                            answer = active_assistant.ask(user_input, session_id)
                            print(f"\nASSISTANT: {answer}")
                            continue
                            
                        intent = "change"
                        if latest_recipes:
                            try:
                                intent = classify_intent(user_input)
                            except Exception:
                                intent = "change"
                        
                        if intent == "select":
                            selected = pick_recipe(user_input, all_shown_recipes, latest_recipes)
                            
                            if selected:
                                recipe_text = json.dumps(selected, indent=2, ensure_ascii=False)
                                active_assistant = RecipeAssistant(recipe_text)
                                print(f"\nASSISTANT: Great choice! You've selected **{selected.get('title')}**. Ask me anything about it.")
                                continue
                            else:
                                intent = "change"
                            
                        if intent == "more":
                            print("\nASSISTANT: (Feature placeholder: Fetching more recipes...)")
                            continue
                            
                        if intent == "change":
                            search_history.append(user_input)
                            combined_query = " ".join(search_history)
                            
                            result = run_agent_full(combined_query)
                            
                            if "error" in result:
                                print(f"\nASSISTANT: {result['error']}")
                                search_history.pop() # Remove the bad query so it doesn't poison the history
                            else:
                                latest_recipes = result.get("recipes", [])
                                
                                existing_titles = {r.get("title") for r in all_shown_recipes}
                                for r in latest_recipes:
                                    if r.get("title") not in existing_titles:
                                        all_shown_recipes.append(r)
                                        
                                print(f"\nASSISTANT: {result.get('summary')}")
                            
                    except KeyboardInterrupt:
                        print("\nExiting agent...")
                        break
                
            elif intent == "more":
                print("\nASSISTANT: (Feature placeholder: Fetching more recipes...)")
                continue
                
            search_history.append(user_input)
            combined_query = " ".join(search_history)
            
            result = run_agent_full(combined_query)
            
            if "error" in result:
                print(f"\nASSISTANT: {result['error']}")
                search_history.pop()
            else:
                latest_recipes = result.get("recipes", [])
                
                existing_titles = {r.get("title") for r in all_shown_recipes}
                for r in latest_recipes:
                    if r.get("title") not in existing_titles:
                        all_shown_recipes.append(r)
                        
                print(f"\nASSISTANT: {result.get('summary')}")
                
        except KeyboardInterrupt:
            print("\nExiting agent...")
            break