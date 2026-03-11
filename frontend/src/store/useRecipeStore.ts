import type { Prompt, ChatPrompt, SearchResponse } from "@/types/type";
import { create } from "zustand";

interface RecipeStore {
  prompt: Prompt;
  initialPrompt: Prompt;
  initialRes: SearchResponse;
  sessionId: string;
  recipeText: string;

  setPrompt: (prompt: Prompt) => void;
  setInitialPrompt: (initialPrompt: Prompt) => void;
  clearInitialPrompt: () => void;
  setInitialRes: (initialRes: SearchResponse) => void;
  setSessionId: (sessionId: string) => void;
  setRecipeText: (recipeText: string) => void;
  clearRecipeText: () => void;

  // Handle the API request and response
  createChatSession: (newPrompt: Prompt) => Promise<Response>;
  fetchChat: (chatPrompt: ChatPrompt) => Promise<Response>;
}

export const useRecipeStore = create<RecipeStore>((set) => ({
  prompt: { promptText: "" },
  initialPrompt: { promptText: "" },
  initialRes: {
    summary: "",
    session_id: "",
  },
  sessionId: "",
  recipeText: "",

  setPrompt: (prompt: Prompt) => set({ prompt }),
  setInitialPrompt: (initialPrompt: Prompt) => set({ initialPrompt }),
  clearInitialPrompt: () => set({ initialPrompt: { promptText: "" } }),
  setInitialRes: (initialRes: SearchResponse) => set({ initialRes }),
  setSessionId: (sessionId: string) => set({ sessionId }),
  setRecipeText: (recipeText: string) => set({ recipeText }),
  clearRecipeText: () => set({ recipeText: "" }),

  // POST: send prompt to FastAPI for recipe search
  createChatSession: async (newPrompt: Prompt) => {
    const res = await fetch("/api/search", {
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

  // POST: handle the following conversation and get chat response payload
  fetchChat: async (chatPrompt: ChatPrompt) => {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: chatPrompt.chatSessionId,
        message: chatPrompt.message,
        recipe_text: chatPrompt.recipeText,
      }),
    });

    return res;
  },
}));
