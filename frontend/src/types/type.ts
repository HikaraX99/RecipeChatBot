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

export type RecipeCard = {
  id: string;
  title: string;
  image_url: string;
};

// searching a new recipe
export type SearchResponse = {
  summary: string;
  session_id: string;
  recipe_cards: RecipeCard[];
};

// Chat response
export type ChatResponse = {
  answer: string;
  recipe_cards: RecipeCard[];
  recipe_text?: string | null;
};
