"use client";

import { useState, useCallback } from "react";
import type { ChatMessage } from "@/lib/api";
import { FileUpload } from "@/components/FileUpload";
import { Sidebar, type ChatItem } from "@/components/Sidebar";
import { Chatbox } from "@/components/Chatbox";

export default function Dashboard() {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messagesByChatId, setMessagesByChatId] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [fileNameByChatId, setFileNameByChatId] = useState<
    Record<string, string>
  >({});

  const handleUpload = useCallback((file: File) => {
    const id = crypto.randomUUID();
    const title = file.name || "New chat";
    setChats((prev) => [
      ...prev,
      { id, title, createdAt: new Date() },
    ]);
    setMessagesByChatId((prev) => ({ ...prev, [id]: [] }));
    setFileNameByChatId((prev) => ({ ...prev, [id]: file.name }));
    setActiveChatId(id);
  }, []);

  const handleNewChat = useCallback(() => {
    setActiveChatId(null);
  }, []);

  const setMessagesForActive = useCallback(
    (updater: React.SetStateAction<ChatMessage[]>) => {
      if (activeChatId === null) return;
      setMessagesByChatId((prev) => ({
        ...prev,
        [activeChatId]:
          typeof updater === "function" ? updater(prev[activeChatId] ?? []) : updater,
      }));
    },
    [activeChatId]
  );

  const activeMessages = activeChatId ? messagesByChatId[activeChatId] ?? [] : [];
  const activeFileName = activeChatId ? fileNameByChatId[activeChatId] : undefined;

  return (
    <div className="flex min-h-0 flex-1">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={handleNewChat}
      />
      <main className="flex min-w-0 flex-1 flex-col items-center justify-start overflow-auto p-6">
        {activeChatId === null ? (
          <div className="w-full max-w-2xl">
            <FileUpload onUpload={handleUpload} />
          </div>
        ) : (
          <div className="w-full max-w-2xl">
            <Chatbox
              messages={activeMessages}
              setMessages={setMessagesForActive}
              conversationId={activeChatId}
              fileName={activeFileName}
            />
          </div>
        )}
      </main>
    </div>
  );
}
