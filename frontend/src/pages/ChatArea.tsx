import * as React from "react";
import ChatInput from "@/components/ChatInput";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type {
  ActionState,
  Prompt,
  SearchResponse,
  ChatPrompt,
  ChatResponse,
} from "@/types/type";
import { toast } from "sonner";
import { useRecipeStore } from "@/store/useRecipeStore";
import { RefreshCw } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";

type Role = "user" | "assistant";

type Msg = {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
};

const STARTER_MESSAGE = "What kind of recipe are you looking for?";

function createStarterMessage(): Msg {
  return {
    id: uid(),
    role: "assistant",
    content: STARTER_MESSAGE,
    createdAt: Date.now(),
  };
}

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function formatTime(ts: number) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const ChatArea = () => {
  // set initial message object
  const [messages, setMessages] = React.useState<Msg[]>(() => [createStarterMessage()]);

  // get the initial prompt
  const initialPrompt = useRecipeStore((s) => s.initialPrompt);
  const clearInitialPrompt = useRecipeStore((s) => s.clearInitialPrompt);

  // Get the functions from RecipeStore
  const {
    createChatSession,
    fetchChat,
    setSessionId,
    setPrompt,
    setRecipeText,
  } = useRecipeStore();

  const [input, setInput] = React.useState("");
  const [isThinking, setIsThinking] = React.useState(false);

  const bottomRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isThinking]);

  // handle the prompt send and agent reply
  const assistantReply = React.useCallback(
    async (newPrompt: Prompt) => {
      setIsThinking(true);

      // Recipe search response object
      let searchRes: SearchResponse = {
        summary: "",
        session_id: "",
      };

      let chatRes: ChatResponse = {
        answer: "",
      };

      let replyText = "";

      try {
        const currentSessionId = useRecipeStore.getState().sessionId;
        // if currentSessionId does not exist
        if (!currentSessionId) {
          // response from the recipe search
          const sessionRes = await createChatSession(newPrompt);

          if (sessionRes.ok) {
            searchRes = await sessionRes.json();
            replyText = searchRes.summary;
            setSessionId(searchRes.session_id);
            setRecipeText(searchRes.summary);
          } else {
            replyText = "ERROR: Please check your payload.";
          }
        } else {
          const currentRecipe = useRecipeStore.getState().recipeText;
          // set the chant prompt
          const chatPrompt: ChatPrompt = {
            chatSessionId: currentSessionId,
            message: newPrompt.promptText,
            recipeText: currentRecipe,
          };

          // wait for the LLM
          const chatBotRes = await fetchChat(chatPrompt);
          if (chatBotRes.ok) {
            chatRes = await chatBotRes.json();
            replyText = chatRes.answer;
          } else {
            replyText = `ERROR: Please check your session ID and payload.`;
          }
        }
        const reply: Msg = {
          id: uid(),
          role: "assistant",
          content: replyText,
          createdAt: Date.now(),
        };

        setMessages((prev) => [...prev, reply]);
      } catch (err) {
        const reply: Msg = {
          id: uid(),
          role: "assistant",
          content: `ERROR: Request failed; ${(err as Error).message}`,
          createdAt: Date.now(),
        };

        setMessages((prev) => [...prev, reply]);
      } finally {
        setIsThinking(false);
      }
    },
    [
      createChatSession,
      fetchChat,
      setRecipeText,
      setSessionId,
    ],
  );

  const refreshRecipeSearch = React.useCallback(() => {
    if (isThinking) return;

    const { setPrompt, setSessionId, setRecipeText } = useRecipeStore.getState();

    setPrompt({ promptText: "" });
    setSessionId("");
    setRecipeText("");
    setInput("");
    setMessages([createStarterMessage()]);
  }, [isThinking]);

  // function handle the prompt send
  const sendPrompt = React.useCallback(
    async (text: string) => {
      // crate a new prompt object
      const newPrompt: Prompt = { promptText: text };

      setPrompt(newPrompt);

      const userMsg: Msg = {
        id: uid(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      };

      setMessages((prev) => [...prev, userMsg]);

      // send prompt object to the agent
      assistantReply(newPrompt);

      // clean the user input
      setInput("");
    },
    [assistantReply, setPrompt],
  );

  // form initial state
  const initialState: ActionState = { error: null };

  // handle the user action, send the user input to the store
  const handlePromptAction = async (
    _prev: ActionState,
    payload: FormData,
  ): Promise<ActionState> => {
    try {
      const text = String(payload.get("promptText")).trim();

      if (!text) return { error: "Please enter a message." };
      if (isThinking) return { error: "Please wait for the current response." };

      // handle the prompt
      await sendPrompt(text);

      return { error: null };
    } catch (err) {
      toast.error("Failed to generate response");
      return { error: (err as Error).message };
    }
  };

  // user input form action
  const hasAutoSentRef = React.useRef(false);
  const [state, formAction, isPending] = React.useActionState<
    ActionState,
    FormData
  >(handlePromptAction, initialState);

  // handle the initial prompt
  React.useEffect(() => {
    if (hasAutoSentRef.current) return;
    if (!initialPrompt?.promptText) return;

    hasAutoSentRef.current = true;
    sendPrompt(initialPrompt.promptText);
    clearInitialPrompt();
  }, [initialPrompt, clearInitialPrompt, sendPrompt]);

  // error toaster
  React.useEffect(() => {
    if (state?.error) toast.error(state.error);
  }, [state?.error]);

  return (
    <div className="h-full bg-background">
      <div className="mx-auto flex h-full flex-col">
        {/* Messages (scroll area) */}
        {/* Main column */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="px-4 pt-4 sm:px-6">
            <div className="mx-auto max-w-4xl flex flex-col gap-5">
              {messages.map((m) => (
                <MessageRow
                  key={m.id}
                  msg={m}
                  onRefresh={m.role === "assistant" ? refreshRecipeSearch : undefined}
                  isRefreshing={isThinking}
                />
              ))}

              {isThinking && <ThinkingRow />}

              <div ref={bottomRef} />
            </div>
          </div>
        </ScrollArea>

        <div className="mt-auto shrink-0 bg-background/80 backdrop-blur">
          <div className="mx-auto max-w-4xl px-4 pt-2 sm:px-6">
            <ChatInput
              value={input}
              onChange={setInput}
              formAction={formAction}
              disabled={isPending}
            />
            <p className="mt-2 mb-2 text-center text-xs text-muted-foreground">
              Enter to send • Shift+Enter for newline
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// message column
function MessageRow({
  msg,
  onRefresh,
  isRefreshing,
}: {
  msg: Msg;
  onRefresh?: () => void;
  isRefreshing: boolean;
}) {
  const isUser = msg.role === "user";

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "flex max-w-[90%] items-end gap-3",
          isUser && "flex-row-reverse",
        )}
      >
        {/* Avatar */}
        <Avatar className="h-8 w-8">
          <AvatarFallback className="text-[10px]">
            {isUser ? "You" : "AI"}
          </AvatarFallback>
        </Avatar>

        {isUser ? (
          <div className="rounded-2xl bg-primary px-4 py-3 text-base leading-relaxed text-primary-foreground shadow-sm">
            <div className="whitespace-pre-wrap">{msg.content}</div>
            <div className="mt-2 text-[11px] opacity-70">
              {formatTime(msg.createdAt)}
            </div>
          </div>
        ) : (
          <div>
            <div className="rounded-2xl bg-gray-100 px-4 py-3 text-base leading-relaxed text-primary-foreground shadow-sm">
              <div className="whitespace-pre-wrap text-base leading-relaxed text-foreground">
                {msg.content}
              </div>
              <div className="mt-2 text-[11px] text-muted-foreground">
                {formatTime(msg.createdAt)}
              </div>
            </div>
            <div className="pt-1">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7 cursor-pointer rounded-sm bg-transparent p-0 hover:bg-gray-100"
                      onClick={onRefresh}
                      disabled={!onRefresh || isRefreshing}
                    >
                      <RefreshCw className="size-4.5 text-gray-600" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Try different recipes</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const ThinkingRow = () => {
  return (
    <div className="flex justify-start">
      <div className="flex max-w-[90%] items-end gap-3">
        <Avatar className="h-8 w-8">
          <AvatarFallback className="text-[10px]">AI</AvatarFallback>
        </Avatar>

        <div className="px-1">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="inline-flex gap-1">
              <Dot />
              <Dot className="animation-delay-150" />
              <Dot className="animation-delay-300" />
            </span>
            <span>Thinking…</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const Dot = ({ className = "" }: { className?: string }) => {
  return (
    <span
      className={`h-2 w-2 animate-bounce rounded-full bg-foreground/40 ${className}`}
    />
  );
};

export default ChatArea;
