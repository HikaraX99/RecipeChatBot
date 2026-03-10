import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from search import search_recipes

class Ingredient(BaseModel):
    name: str = Field(description="The food item name (e.g., 'carrot', 'corn')")
    quantity_g: Optional[float] = Field(
        default=None,
        description=(
            "Weight in grams, or null if the user gave no quantity. "
            "ONLY set a number when the user explicitly stated an amount "
            "(e.g. '200g of corn', '2 chicken breasts'). "
            "If the user says 'I have broccoli' with no amount → null. "
            "DO NOT guess or estimate a quantity."
        )
    )

class RecipeSearchParams(BaseModel):
    query_text: str = Field(
        description=(
            "Keywords for the recipe database search. "
            "ONLY include ingredient names or dish names — nothing else. "
            "Strip out words like 'recipe', 'high protein', 'healthy', 'light', 'vegan', 'quick', etc. "
            "These are nutrition/diet descriptors that won't match anything in the database. "
            "CORRECT examples: 'chicken', 'corn salsa', 'banana muffins', 'broccoli stir fry'. "
            "WRONG examples: 'high protein chicken recipe', 'healthy vegan dessert', 'light meal ideas'."
        )
    )

    # --- Calorie range ---
    min_calories: Optional[float] = Field(
        default=None,
        description="Minimum calories per serving, or null if not specified."
    )
    max_calories: Optional[float] = Field(
        default=None,
        description=(
            "Upper calorie limit per serving, or null. "
            "Extract from explicit mentions like 'under 500 calories'. "
            "If the user says 'light meal' use 500. If 'hungry' or 'big meal' use 1000. "
            "Use null if no calorie constraint is implied."
        )
    )

    # --- Protein ---
    min_protein: Optional[float] = Field(
        default=None,
        description=(
            "Minimum protein in grams per serving, or null. "
            "Use 30 if user mentions strength training or muscle gain. "
            "Use 20 for general fitness goals. "
            "Use null if protein is not relevant."
        )
    )
    max_protein: Optional[float] = Field(
        default=None,
        description="Maximum protein in grams per serving, or null."
    )

    # --- Fat ---
    min_fat: Optional[float] = Field(
        default=None,
        description="Minimum total fat in grams per serving, or null."
    )
    max_fat: Optional[float] = Field(
        default=None,
        description="Maximum total fat in grams per serving, or null."
    )

    # --- Saturated fat ---
    min_saturated_fat: Optional[float] = Field(
        default=None,
        description="Minimum saturated fat in grams per serving, or null."
    )
    max_saturated_fat: Optional[float] = Field(
        default=None,
        description="Maximum saturated fat in grams per serving, or null."
    )

    # --- Carbohydrates ---
    min_carbs: Optional[float] = Field(
        default=None,
        description="Minimum carbohydrates in grams per serving, or null."
    )
    max_carbs: Optional[float] = Field(
        default=None,
        description="Maximum carbohydrates in grams per serving, or null."
    )

    # --- Fiber ---
    min_fiber: Optional[float] = Field(
        default=None,
        description="Minimum dietary fiber in grams per serving, or null."
    )
    max_fiber: Optional[float] = Field(
        default=None,
        description="Maximum dietary fiber in grams per serving, or null."
    )

    # --- Sugar ---
    min_sugar: Optional[float] = Field(
        default=None,
        description="Minimum sugar in grams per serving, or null."
    )
    max_sugar: Optional[float] = Field(
        default=None,
        description="Maximum sugar in grams per serving, or null."
    )

    # --- Sodium ---
    min_sodium: Optional[float] = Field(
        default=None,
        description="Minimum sodium in grams per serving, or null."
    )
    max_sodium: Optional[float] = Field(
        default=None,
        description="Maximum sodium in grams per serving, or null."
    )

    # --- Ingredients filter ---
    available_ingredients: Optional[List[Ingredient]] = Field(
        default=None,
        description=(
            "Ingredients the user already has, as a list of objects. "
            "Each object MUST have 'name' (string) and 'quantity_g' (number or null). "
            "Example: [{\"name\": \"corn\", \"quantity_g\": 200}]. "
            "Do NOT use a flat dict like {\"corn\": 200}."
        )
    )

    # --- Result control ---
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return."
    )

    @field_validator(
        "min_calories", "max_calories",
        "min_protein", "max_protein",
        "min_fat", "max_fat",
        "min_saturated_fat", "max_saturated_fat",
        "min_carbs", "max_carbs",
        "min_fiber", "max_fiber",
        "min_sugar", "max_sugar",
        "min_sodium", "max_sodium",
        mode="before"
    )
    @classmethod
    def coerce_none_string(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("none", "null", ""):
            return None
        return v

    @field_validator("available_ingredients", mode="before")
    @classmethod
    def coerce_ingredient_dict(cls, v):
        if isinstance(v, dict):
            v = [{"name": k, "quantity_g": float(qty)} for k, qty in v.items()]
        if isinstance(v, list):
            INVALID_NAMES = {"null", "none", "n/a", ""}
            v = [
                i for i in v
                if isinstance(i, dict)
                and i.get("name", "").strip().lower() not in INVALID_NAMES
            ]
            return v if v else None
        return v


@tool(args_schema=RecipeSearchParams)
def find_recipes(
    query_text: str,
    min_calories: Optional[float] = None,
    max_calories: Optional[float] = None,
    min_protein: Optional[float] = None,
    max_protein: Optional[float] = None,
    min_fat: Optional[float] = None,
    max_fat: Optional[float] = None,
    min_saturated_fat: Optional[float] = None,
    max_saturated_fat: Optional[float] = None,
    min_carbs: Optional[float] = None,
    max_carbs: Optional[float] = None,
    min_fiber: Optional[float] = None,
    max_fiber: Optional[float] = None,
    min_sugar: Optional[float] = None,
    max_sugar: Optional[float] = None,
    min_sodium: Optional[float] = None,
    max_sodium: Optional[float] = None,
    available_ingredients: Optional[List[Ingredient]] = None,
    max_results: int = 10,
):
    """
    Search the recipe database using parameters extracted from the user's request.
    Always call this tool when the user wants recipe suggestions.
    """
    print("\n--- RECIPE SEARCH TRIGGERED ---")
    print(f"  Query          : {query_text}")
    print(f"  Cal range      : {min_calories} – {max_calories} kcal")
    print(f"  Protein range  : {min_protein} – {max_protein} g")
    print(f"  Fat range      : {min_fat} – {max_fat} g")
    print(f"  Sat fat range  : {min_saturated_fat} – {max_saturated_fat} g")
    print(f"  Carbs range    : {min_carbs} – {max_carbs} g")
    print(f"  Fiber range    : {min_fiber} – {max_fiber} g")
    print(f"  Sugar range    : {min_sugar} – {max_sugar} g")
    print(f"  Sodium range   : {min_sodium} – {max_sodium} g")
    print(f"  Max results    : {max_results}")
    if available_ingredients:
        pantry_display = [
            {"name": i.name, "quantity_g": i.quantity_g}
            for i in available_ingredients
        ]
        print(f"  Pantry         : {pantry_display}")

    # Derive ingredient name list for search.py's `ingredients` filter
    ingredient_names: Optional[List[str]] = None
    if available_ingredients:
        ingredient_names = [i.name for i in available_ingredients if i.name.strip()]

    results_raw = search_recipes.invoke({
        "title": query_text,
        "ingredients": ingredient_names,
        "min_calories": min_calories,
        "max_calories": max_calories,
        "min_protein": min_protein,
        "max_protein": max_protein,
        "min_fat": min_fat,
        "max_fat": max_fat,
        "min_saturated_fat": min_saturated_fat,
        "max_saturated_fat": max_saturated_fat,
        "min_carbs": min_carbs,
        "max_carbs": max_carbs,
        "min_fiber": min_fiber,
        "max_fiber": max_fiber,
        "min_sugar": min_sugar,
        "max_sugar": max_sugar,
        "min_sodium": min_sodium,
        "max_sodium": max_sodium,
        "max_results": max_results,
    })

    if not results_raw:
        return "No recipes found matching those criteria."

    return json.dumps(results_raw) if isinstance(results_raw, list) else results_raw


SYSTEM_PROMPT = """You are a helpful nutrition and cooking assistant.

When a user asks for recipe ideas:
1. Call the `find_recipes` tool with parameters extracted from their message.
2. Extract all numeric nutrition constraints from context clues, not just explicit numbers.
3. List the user's available ingredients in `available_ingredients` as a list of objects.
4. After receiving tool results, you MUST immediately present the recipes to the user. Do NOT ask follow-up questions. Do NOT say you need more information. Just list the recipes.

How to present results:
- List each recipe by its "title" field.
- Include calories (nutrition.calories) and protein (nutrition.protein_g) per serving.
- Keep each entry to 2-3 lines.
- Example format:
  🍗 Grilled Chicken Salad — 320 kcal | 35g protein
  A light, fresh salad with grilled chicken breast and mixed greens.

Parameter extraction rules:
- "under X calories" → max_calories = X
- "light / healthy" → max_calories = 500
- "hungry / big meal" → max_calories = 1000
- "post workout / strength training / leg day" → min_protein = 30
- "high protein" → min_protein = 25
- "low fat" → max_fat = 10
- "low carb / keto" → max_carbs = 20
- "high fiber" → min_fiber = 8
- Available ingredients → available_ingredients list, e.g. [{"name": "corn", "quantity_g": 200}]
- Query text MUST only contain ingredient or dish names, e.g. "chicken" or "corn taco".

CRITICAL RULES:
1. You MUST present the recipes returned by `find_recipes`. Never ignore tool results.
2. If the tool returns an empty list or "No recipes found", tell the user nothing matched.
3. NEVER invent recipes or use your own knowledge. Only use what the tool returns.
4. NEVER ask clarifying questions before calling the tool — call it immediately.
"""

def run_agent(user_input: str) -> str:
    llm = ChatOllama(model="llama3.1", temperature=0)
    tools = [find_recipes]
    llm_with_tools = llm.bind_tools(tools)  # used for tool-call turn only
    llm_plain = llm                          # used for summary turn — no tool overhead
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = response.tool_calls

        if not tool_calls and response.content:
            try:
                raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_calls = [{
                        "name": parsed["name"],
                        "args": parsed.get("parameters") or parsed.get("args", {}),
                        "id": "fallback-0"
                    }]
            except (json.JSONDecodeError, KeyError):
                pass

        if not tool_calls:
            return response.content

        all_results = []
        for call in tool_calls:
            tool_fn = tool_map.get(call["name"])
            if tool_fn:
                result = tool_fn.invoke(call["args"])
                if isinstance(result, (list, dict)):
                    tool_content = json.dumps(result)
                else:
                    tool_content = str(result)
                messages.append(ToolMessage(content=tool_content, tool_call_id=call["id"]))
                all_results.append(tool_content)

        # Once we have tool results, build one summary prompt and return immediately.
        # Do NOT loop again — that's what caused the repetition.
        if all_results:
            combined = "\n".join(all_results)

            # Parse and deduplicate by title before handing to the LLM
            try:
                recipes = json.loads(combined)
                seen = set()
                unique = []
                for r in recipes:
                    t = r.get("title", "")
                    if t and t not in seen:
                        seen.add(t)
                        unique.append(r)
                combined = json.dumps(unique)
            except (json.JSONDecodeError, TypeError):
                pass

            # Reuse the existing message thread — avoids a second full LLM call
            messages.append(HumanMessage(
                content=(
                    "Here are the results from the recipe database:\n"
                    f"{combined}\n\n"
                    "Summarise ONLY the unique recipes above — do not repeat any title. "
                    "For each recipe show: title, calories, protein, and one sentence description. "
                    "Do NOT invent recipes. Do NOT ask follow-up questions."
                )
            ))
            # Use plain llm (no tools) so it just generates text, no tool-call overhead
            final = llm_plain.invoke(messages)
            return final.content

    return "Agent loop limit reached without a final answer."


if __name__ == "__main__":
    print("Recipe Agent started. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nUSER: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("Exiting agent...")
                break

            if not user_input.strip():
                continue

            print(f"\n")

            answer = run_agent(user_input)

            print(f"\nASSISTANT: {answer}\n")

        except KeyboardInterrupt:
            print("\nExiting agent...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")