// Form action type
export type ActionState = { error: string | null };

// Prompt input
export type Prompt = {
  promptText: string;
};

// chat response
export type ChatResponse = {
  answer: string;
  query: string;
};
