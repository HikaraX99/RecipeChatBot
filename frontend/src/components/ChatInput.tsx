import * as React from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send } from "lucide-react";

type ChatInputProps = {
  value: string;
  onChange: (next: string) => void;
  formAction: (payload: FormData) => void;
  disabled?: boolean;
  placeholder?: string;
  onAdd?: () => void;
};

// Chat input UI which handle the form action 
const ChatInput = ({
  value,
  onChange,
  formAction,
  disabled,
  placeholder = "Ask recipes",
}: ChatInputProps) => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter makes newline
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      e.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <div className="w-full flex justify-center">
      <form
        action={formAction}
        className="w-full max-w-3xl rounded-[28px] border bg-background shadow-sm px-2 py-2"
      >
        <div className="flex items-center flex-col">
          <Textarea
            name="promptText"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="border-0 min-h-10 max-h-64 resize-none shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 text-base!"
            disabled={disabled}
          />

          <div className="flex w-full justify-end">
            <Button
              type="submit"
              className="h-10 w-10 rounded-full cursor-pointer"
              aria-label="Send"
              disabled={disabled}
            >
              <Send className="h-8 w-8" />
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;
