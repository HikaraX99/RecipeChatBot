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

extract_chain = EXTRACT_PROMPT | llm
summary_chain = SUMMARY_PROMPT | llm


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


if __name__ == "__main__":
    print("Recipe Agent started. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nUSER: ").strip()
            if user_input.lower() in ("quit", "exit"):
                print("Exiting agent...")
                break
            if not user_input:
                continue

            answer = run_agent(user_input)
            print(f"\nASSISTANT: {answer}\n")

        except KeyboardInterrupt:
            print("\nExiting agent...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")