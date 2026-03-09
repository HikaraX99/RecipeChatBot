from typing import List, Optional, Type
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from elasticsearch import Elasticsearch


# ── Input Schema ──────────────────────────────────────────────────────────────
# Flat schema — much easier for LLMs to populate reliably via function calling.

class RecipeSearchInput(BaseModel):
    """Search the recipe database. All fields are optional."""

    # --- Text filters ---
    title: Optional[str] = Field(None, description="Partial recipe title to search for.")
    ingredients: Optional[List[str]] = Field(
        None,
        description="Ingredients that MUST appear in the recipe, e.g. ['chicken', 'garlic'].",
    )

    # --- Calorie range ---
    min_calories: Optional[float] = Field(None, description="Minimum calories per serving.")
    max_calories: Optional[float] = Field(None, description="Maximum calories per serving.")

    # --- Protein ---
    min_protein: Optional[float] = Field(None, description="Minimum protein in grams per serving.")
    max_protein: Optional[float] = Field(None, description="Maximum protein in grams per serving.")

    # --- Fat ---
    min_fat: Optional[float] = Field(None, description="Minimum total fat in grams per serving.")
    max_fat: Optional[float] = Field(None, description="Maximum total fat in grams per serving.")

    # --- Saturated fat ---
    min_saturated_fat: Optional[float] = Field(None, description="Minimum saturated fat in grams.")
    max_saturated_fat: Optional[float] = Field(None, description="Maximum saturated fat in grams.")

    # --- Carbohydrates ---
    min_carbs: Optional[float] = Field(None, description="Minimum carbohydrates in grams per serving.")
    max_carbs: Optional[float] = Field(None, description="Maximum carbohydrates in grams per serving.")

    # --- Fiber ---
    min_fiber: Optional[float] = Field(None, description="Minimum dietary fiber in grams per serving.")
    max_fiber: Optional[float] = Field(None, description="Maximum dietary fiber in grams per serving.")

    # --- Sugar ---
    min_sugar: Optional[float] = Field(None, description="Minimum sugar in grams per serving.")
    max_sugar: Optional[float] = Field(None, description="Maximum sugar in grams per serving.")

    # --- Sodium (stored as grams in this DB) ---
    min_sodium: Optional[float] = Field(None, description="Minimum sodium in grams per serving.")
    max_sodium: Optional[float] = Field(None, description="Maximum sodium in grams per serving.")

    # --- Result control ---
    max_results: int = Field(10, description="Maximum number of results to return.")


# ── Elasticsearch query builder ───────────────────────────────────────────────

# Maps (min_field, max_field) in the schema → ES document path
NUTRIENT_RANGE_MAP = {
    ("min_calories",      "max_calories"):      "nutrition.calories",
    ("min_protein",       "max_protein"):       "nutrition.proteinContent",
    ("min_fat",           "max_fat"):           "nutrition.fatContent",
    ("min_saturated_fat", "max_saturated_fat"): "nutrition.saturatedFatContent",
    ("min_carbs",         "max_carbs"):         "nutrition.carbohydrateContent",
    ("min_fiber",         "max_fiber"):         "nutrition.fiberContent",
    ("min_sugar",         "max_sugar"):         "nutrition.sugarContent",
    ("min_sodium",        "max_sodium"):        "nutrition.sodiumContent",
}


def build_recipe_query(p: RecipeSearchInput) -> dict:
    must_clauses = []
    filter_clauses = []

    # Full-text: title
    if p.title:
        must_clauses.append({
            "match": {"title": {"query": p.title, "fuzziness": "AUTO"}}
        })

    # Full-text: every requested ingredient must be present
    if p.ingredients:
        for ingredient in p.ingredients:
            must_clauses.append({
                "match": {
                    "ingredients": {
                        "query": ingredient,
                        "fuzziness": "AUTO",
                        "operator": "and",
                    }
                }
            })

    # Numeric range filters
    for (min_field, max_field), es_path in NUTRIENT_RANGE_MAP.items():
        min_val = getattr(p, min_field)
        max_val = getattr(p, max_field)

        range_clause = {}
        if min_val is not None:
            range_clause["gte"] = min_val
        if max_val is not None:
            range_clause["lte"] = max_val
        if range_clause:
            filter_clauses.append({"range": {es_path: range_clause}})

    bool_query: dict = {}
    if must_clauses:
        bool_query["must"] = must_clauses
    if filter_clauses:
        bool_query["filter"] = filter_clauses

    if not bool_query:
        return {"query": {"match_all": {}}, "size": p.max_results}

    return {"query": {"bool": bool_query}, "size": p.max_results}


# ── LangChain Tool ────────────────────────────────────────────────────────────

class RecipeSearchTool(BaseTool):
    name: str = "recipe_search"
    description: str = (
        "Search a recipe database by title, required ingredients, and/or nutritional "
        "ranges (calories, protein, fat, saturated fat, carbohydrates, fiber, sugar, sodium). "
        "All parameters are optional — combine them freely to narrow results."
    )
    args_schema: Type[BaseModel] = RecipeSearchInput

    # PrivateAttr keeps the ES client out of Pydantic's field system entirely,
    # avoiding serialization errors and Pydantic validation issues.
    _es: Elasticsearch = PrivateAttr()
    _index: str = PrivateAttr()

    def __init__(self, es_client: Elasticsearch, index_name: str = "recipes", **kwargs):
        super().__init__(**kwargs)
        self._es = es_client
        self._index = index_name

    def _format_hit(self, hit: dict) -> dict:
        src = hit.get("_source", {})
        n = src.get("nutrition", {})
        return {
            "title":       src.get("title"),
            "servings":    src.get("servings"),
            "total_time":  src.get("total_time"),
            "source_url":  src.get("source_url"),
            "ingredients": src.get("ingredients", []),
            "steps":       src.get("steps", []),
            "nutrition": {
                "calories":        n.get("calories"),
                "protein_g":       n.get("proteinContent"),
                "fat_g":           n.get("fatContent"),
                "saturated_fat_g": n.get("saturatedFatContent"),
                "carbs_g":         n.get("carbohydrateContent"),
                "fiber_g":         n.get("fiberContent"),
                "sugar_g":         n.get("sugarContent"),
                "sodium_g":        n.get("sodiumContent"),
            },
            "score": hit.get("_score"),
        }

    # Explicit kwargs mirror RecipeSearchInput exactly so LangChain routes them correctly
    def _run(
        self,
        title: Optional[str] = None,
        ingredients: Optional[List[str]] = None,
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
        max_results: int = 10,
    ) -> List[dict]:
        params = RecipeSearchInput(
            title=title,
            ingredients=ingredients,
            min_calories=min_calories, max_calories=max_calories,
            min_protein=min_protein,   max_protein=max_protein,
            min_fat=min_fat,           max_fat=max_fat,
            min_saturated_fat=min_saturated_fat, max_saturated_fat=max_saturated_fat,
            min_carbs=min_carbs,       max_carbs=max_carbs,
            min_fiber=min_fiber,       max_fiber=max_fiber,
            min_sugar=min_sugar,       max_sugar=max_sugar,
            min_sodium=min_sodium,     max_sodium=max_sodium,
            max_results=max_results,
        )
        query = build_recipe_query(params)
        response = self._es.search(index=self._index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        return [self._format_hit(h) for h in hits]

    async def _arun(self, **kwargs) -> List[dict]:
        # For true async, swap in AsyncElasticsearch and await the search call
        return self._run(**kwargs)

es = Elasticsearch("http://localhost:9200")
search_recipes = RecipeSearchTool(es)