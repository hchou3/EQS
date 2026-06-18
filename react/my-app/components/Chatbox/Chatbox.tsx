"use client";

import { useState, useCallback } from "react";
import type { ChatMessage } from "@/lib/api";
import { sendMessage as sendMessageApi } from "@/lib/api";

// API_CALL (optional): Load conversation history when opening a thread.
// e.g. useEffect(() => { getConversationHistory(conversationId).then(setMessages); }, [conversationId]);
// See lib/api/chat.ts getConversationHistory().

type ChatboxProps = {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  conversationId?: string;
  fileName?: string;
};

export default function Chatbox({
  messages,
  setMessages,
  conversationId,
  fileName,
}: ChatboxProps) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || isLoading) return;

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");
      setIsLoading(true);

      try {
        // ——— API_CALL: triggered here ———
        // sendMessageApi() calls the chat backend (see lib/api/chat.ts).
        // Replace the implementation in lib/api/chat.ts with your real endpoint.
        const response = await sendMessageApi(text, conversationId);

        if (response.success && response.message) {
          setMessages((prev) => [...prev, response.message]);
        }
      } catch (err) {
        console.error("Chat API error:", err);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Sorry, something went wrong. Please try again.",
            timestamp: new Date().toISOString(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [input, isLoading, setMessages, conversationId]
  );

  return (
    <section
      className="flex flex-col rounded-lg bg-[var(--surface)] shadow"
      aria-label="Chat"
      style={{ border: "1px solid var(--outline)" }}
    >
      <div
        className="px-4 py-3"
        style={{ borderBottom: "1px solid var(--outline)" }}
      >
        <h2 className="text-sm font-medium text-[var(--outline)]">
          Chat
          {fileName ? (
            <span className="ml-2 font-normal text-[var(--outline-muted)]">
              · {fileName}
            </span>
          ) : null}
        </h2>
      </div>

      <div className="flex min-h-[280px] max-h-[420px] flex-1 flex-col overflow-hidden">
        <div className="chat-messages flex-1 space-y-3 overflow-y-auto p-4 min-h-0">
          {messages.length === 0 && (
            <p className="text-center text-sm text-[var(--outline-muted)]">
              Send a message to start the conversation.
            </p>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`rounded-lg px-3 py-2 text-sm ${
                msg.role === "user"
                  ? "ml-8 bg-[var(--surface-elevated)]"
                  : "mr-8 bg-[var(--outline)]/20"
              }`}
              style={{
                border: "1px solid var(--outline)",
              }}
            >
              <span className="font-medium text-[var(--outline)]">
                {msg.role === "user" ? "You" : "Assistant"}:
              </span>{" "}
              <span className="text-[var(--foreground)]">{msg.content}</span>
            </div>
          ))}
          {isLoading && (
            <div
              className="mr-8 rounded-lg px-3 py-2 text-sm bg-[var(--outline)]/20"
              style={{ border: "1px solid var(--outline)" }}
            >
              <span className="text-[var(--outline-muted)]">Thinking…</span>
            </div>
          )}
        </div>

        <form
          onSubmit={handleSubmit}
          className="p-4"
          style={{ borderTop: "1px solid var(--outline)" }}
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message…"
              className="flex-1 rounded-lg bg-[var(--surface-elevated)] px-4 py-2 text-[var(--foreground)] placeholder:text-[var(--outline-muted)] focus:outline-none disabled:opacity-50"
              style={{
                border: "1px solid var(--outline)",
              }}
              disabled={isLoading}
              aria-label="Message input"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="rounded-lg px-4 py-2 text-sm font-medium text-[var(--surface)] bg-[var(--outline)] transition-opacity hover:opacity-90 disabled:opacity-50"
              style={{ border: "1px solid var(--outline)" }}
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
