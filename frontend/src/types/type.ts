// Form action type
export type ActionState = { error: string | null };

// Prompt for recipe search
export type Prompt = {
  promptText: string;
};

// Prompt for conversation
export type ChatPrompt = {
  chatSessionId: string;
  message: string;
  recipeText: string;
};

// searching a new recipe
export type SearchResponse = {
  summary: string;
  session_id: string;
};

// Chat response
export type ChatResponse = {
  answer: string;
};
