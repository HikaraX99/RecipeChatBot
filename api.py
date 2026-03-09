from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import json

from llm_pre_es import run_agent



app = FastAPI()

# fake in-memory storage for recipes
recipe = []

# based Model for recipe input
class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    steps: list[str]

# root
@app.get("/")
def root():
    return {"message": "Recipe ChatBot API is running!"}

# GET all recipes
@app.get("/recipes")
def get_recipes():
    return {"recipes": recipe}

#GET single recipe by recipe id
@app.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: int):
     
    if recipe_id >= len(recipe) or recipe_id < 0:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    return {"recipe": recipe[recipe_id]}  

# POST a new recipe
@app.post("/recipes")
def create_recipe(new_recipe: Recipe):
    recipe.append(new_recipe)
    return {"message": "Recipe created successfully"}   

#update recipe by id
@app.put("/recipes/{recipe_id}")
def update_recipe(recipe_id: int, updated_recipe: Recipe):
    if recipe_id >= len(recipe) or recipe_id < 0:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    recipe[recipe_id] = updated_recipe
    return {"message": "Recipe updated successfully"}


#delete recipe by id
@app.delete("/recipes/{recipe_id}")
def delete_recipe(recipe_id: int):
    if recipe_id >= len(recipe) or recipe_id < 0:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    recipe.pop(recipe_id)
    return {"message": "Recipe deleted successfully"}

# endpoint to run agent
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: ChatRequest):
    answer = run_agent(request.query)

    return {
        "answer": answer, 
        "query": request.query
        }