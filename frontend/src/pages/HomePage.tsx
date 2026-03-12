import ChatInput from "@/components/ChatInput";
import type { ActionState, Prompt } from "@/types/type";
import { useActionState, useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";
import { useRecipeStore } from "@/store/useRecipeStore";

const HomePage = () => {
  const [input, setInput] = useState("");
  const navigate = useNavigate();

  // import the store function
  const { setInitialPrompt } = useRecipeStore();

  // form initial state
  const initialState: ActionState = { error: null };

  // handle the user input
  const handleSendAction = async (
    _prev: ActionState,
    payload: FormData,
  ): Promise<ActionState> => {
    try {
      const initialPrompt: Prompt = {
        promptText: String(payload.get("promptText")).trim(),
      };
      if (!initialPrompt.promptText)
        return { error: "Please enter a message." };

      // set the initial prompt state
      setInitialPrompt(initialPrompt);

      // go to chat page and pass the initial message
      navigate("/chat");

      // clear local input
      setInput("");

      return { error: null };
    } catch (err) {
      toast.error("Failed to generate response");
      return { error: (err as Error).message };
    }
  };

  // user input form action
  const [state, formAction, isPending] = useActionState<ActionState, FormData>(
    handleSendAction,
    initialState,
  );

  // error toaster
  useEffect(() => {
    if (state?.error) toast.error(state.error);
  }, [state?.error]);

  return (
    <div className="min-h-[calc(80vh-64px)] flex items-center justify-center px-4">
      <div className="w-full max-w-5xl flex flex-col items-center gap-10">
        <h1 className="text-center text-4xl font-normal">
          What kind of recipe are you looking for?
        </h1>
        <div className="w-full">
          <ChatInput
            value={input}
            onChange={setInput}
            formAction={formAction}
            disabled={isPending}
          />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
