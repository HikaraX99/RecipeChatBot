import type { Prompt, ChatResponse } from "@/types/type";
import { create } from "zustand";

interface Recipe {
  title: string;
  ingredients: string[];
  steps: string[];
}

interface RecipeStore {
  prompt: Prompt;
  initialPrompt: Prompt;
  chatId: string;

  setPrompt: (prompt: Prompt) => void;
  setInitialPrompt: (initialPrompt: Prompt) => void;
  clearInitialPrompt: () => void;
  setChatId: (chatId: string) => void;

  createChat: (newPrompt: Prompt) => Promise<Response>;
  fetchChat: (chatId: string) => Promise<ChatResponse>;

  // Wiring with the fastAPI
  fetchRecipes: () => Promise<{ recipes: Recipe[] }>;
  fetchRecipeById: (recipeId: number) => Promise<{ recipe: Recipe }>;
  createRecipe: (newRecipe: Recipe) => Promise<Response>;
  updateRecipe: (recipeId: number, updatedRecipe: Recipe) => Promise<Response>;
  deleteRecipe: (recipeId: number) => Promise<Response>;
}

export const useRecipeStore = create<RecipeStore>((set) => ({
  prompt: { promptText: "" },
  initialPrompt: { promptText: "" },
  chatId: "",

  setPrompt: (prompt: Prompt) => set({ prompt }),
  setInitialPrompt: (initialPrompt: Prompt) => set({ initialPrompt }),
  clearInitialPrompt: () => set({ initialPrompt: { promptText: "" } }),
  setChatId: (chatId: string) => set({ chatId }),

  // POST: send prompt to FastAPI chat endpoint
  createChat: async (newPrompt: Prompt) => {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: newPrompt.promptText,
      }),
    });

    return res;
  },

  // POST: get chat response payload
  fetchChat: async (chatId: string) => {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        // TODO: may need change
        query: chatId,
      }),
    });

    const payload = await res.json();

    return payload;
  },

  // GET: load all recipes
  fetchRecipes: async () => {
    const res = await fetch("/api/recipes", {
      method: "GET",
    });

    const payload = await res.json();

    return payload;
  },

  // GET: load single recipe by id
  fetchRecipeById: async (recipeId: number) => {
    const res = await fetch(`/api/recipes/${recipeId}`, {
      method: "GET",
    });

    const payload = await res.json();

    return payload;
  },

  // POST: create recipe
  createRecipe: async (newRecipe: Recipe) => {
    const res = await fetch("/api/recipes", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(newRecipe),
    });

    return res;
  },

  // PUT: update recipe by id
  updateRecipe: async (recipeId: number, updatedRecipe: Recipe) => {
    const res = await fetch(`/api/recipes/${recipeId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedRecipe),
    });

    return res;
  },

  // DELETE: delete recipe by id
  deleteRecipe: async (recipeId: number) => {
    const res = await fetch(`/api/recipes/${recipeId}`, {
      method: "DELETE",
    });

    return res;
  },
}));
