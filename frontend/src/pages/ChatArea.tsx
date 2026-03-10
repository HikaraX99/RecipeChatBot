import * as React from "react";
import ChatInput from "@/components/ChatInput";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { ActionState, Prompt, ChatResponse } from "@/types/type";
import { toast } from "sonner";
import { useRecipeStore } from "@/store/useRecipeStore";

type Role = "user" | "assistant";

type Msg = {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
};

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function formatTime(ts: number) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const ChatArea = () => {
  const [messages, setMessages] = React.useState<Msg[]>(() => [
    {
      id: uid(),
      role: "assistant",
      content: "Hi! Ask me anything",
      createdAt: Date.now(),
    },
  ]);

  // get the initial prompt
  const initialPrompt = useRecipeStore((s) => s.initialPrompt);
  const clearInitialPrompt = useRecipeStore((s) => s.clearInitialPrompt);

  // Get the functions in RecipeStore
  const { createChat } = useRecipeStore();

  const [input, setInput] = React.useState("");
  const [isTyping, setIsTyping] = React.useState(false);

  const bottomRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isTyping]);

  const assistantReply = async (newPrompt: Prompt) => {
    setIsTyping(true);

    const chatRes = await createChat(newPrompt);
    const payload: ChatResponse = await chatRes.json()
    let replay:string = "";

    if (chatRes.ok){
      replay = payload.answer
    } else {
      replay = "Internal Server Error"
    }

    const reply: Msg = {
      id: uid(),
      role: "assistant",
      content: replay,
      createdAt: Date.now(),
    };

    setMessages((prev) => [...prev, reply]);
    setIsTyping(false);
  };

  // form initial state
  const initialState: ActionState = { error: null };

  // handle the user action, send the user input to the store
  const handlePromptAction = async (
    _prev: ActionState,
    payload: FormData,
  ): Promise<ActionState> => {
    try {
      const text = String(payload.get("promptText")).trim();
      // crate a new prompt object
      const newPrompt: Prompt = {
        promptText: text,
      };

      if (!text || isTyping) return { error: "Please enter a message." };

      const userMsg: Msg = {
        id: uid(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      };

      setMessages((prev) => [...prev, userMsg]);

      // send prompt to the agent
      await assistantReply(newPrompt);

      // clean the user input
      setInput("");

      return { error: null };
    } catch (err) {
      toast.error("Failed to generate response");
      return { error: (err as Error).message };
    }
  };

  // user input form action
  const [state, formAction, isPending] = React.useActionState<
    ActionState,
    FormData
  >(handlePromptAction, initialState);

  // handle the initial prompt
  React.useEffect(() => {
    if (!initialPrompt?.promptText) return;

    const text = initialPrompt.promptText;
    // 1) set input
    setInput(text);

    // 2) send it to form
    const fd = new FormData();
    fd.set("promptText", text);
    formAction(fd);

    clearInitialPrompt();
    window.history.replaceState({}, "");
  }, [formAction, initialPrompt, clearInitialPrompt]);

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
                <MessageRow key={m.id} msg={m} />
              ))}

              {isTyping && <TypingRow />}

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

function MessageRow({ msg }: { msg: Msg }) {
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
          <div className="px-1">
            <div className="whitespace-pre-wrap text-base leading-relaxed text-foreground">
              {msg.content}
            </div>
            <div className="mt-2 text-[11px] text-muted-foreground">
              {formatTime(msg.createdAt)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TypingRow() {
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
            <span>Typing…</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Dot({ className = "" }: { className?: string }) {
  return (
    <span
      className={`h-2 w-2 animate-bounce rounded-full bg-foreground/40 ${className}`}
    />
  );
}

export default ChatArea;
