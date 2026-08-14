"use client";

import { useState, useCallback } from "react";
import CircularDotsLoader from "@/components/CircularDotsLoader";
import type { ChatMessage } from "@/lib/api";
import { FileUpload } from "@/components/FileUpload";
import { Sidebar, type ChatItem } from "@/components/Sidebar";
import { Chatbox } from "@/components/Chatbox";
import AnalysisTabs from "@/components/AnalysisTabs/AnalysisTabs";
import type { CSVUploadResponse } from "@/lib/api/types";

export default function Dashboard() {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messagesByChatId, setMessagesByChatId] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [fileNameByChatId, setFileNameByChatId] = useState<
    Record<string, string>
  >({});
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResult, setAnalysisResult] =
    useState<CSVUploadResponse | null>(null);

  const handleUpload = useCallback(
    async (file: File, llm: string, apiKey?: string) => {
      const id = crypto.randomUUID();
      const title = file.name || "New chat";

      // Create chat entry immediately
      setChats((prev) => [...prev, { id, title, createdAt: new Date() }]);
      setMessagesByChatId((prev) => ({ ...prev, [id]: [] }));
      setFileNameByChatId((prev) => ({ ...prev, [id]: file.name }));
      setActiveChatId(id);
      setIsUploading(true);
      // Clear any previous analysis results
      setAnalysisResult(null);

      try {
        const formData = new FormData();
        formData.append("file", file);

        // Normalize the LLM value to provider name (e.g., "groq/llama3" -> "groq")
        const provider = llm.includes("/") ? llm.split("/")[0] : llm;
        if (provider) formData.append("llm_provider", provider);

        // DEBUG: Log what we're actually sending
        console.log(
          "Original llm:",
          llm,
          "Normalized provider:",
          provider,
          "Has apiKey:",
          !!apiKey,
        );

        if (apiKey) formData.append("llm_api_key", apiKey);

        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/csv/upload`,
          {
            method: "POST",
            headers: {
              Accept: "application/json",
            },
            body: formData,
          },
        );

        if (!response.ok) {
          const errorData = await response
            .json()
            .catch(() => ({ detail: "Upload failed" }));
          throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        // Store analysis results in state for the dashboard component
        // NOTE: We store analysis result separately from chat messages to avoid polluting the chat interface
        setAnalysisResult(data);
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error";
        setMessagesByChatId((prev) => ({
          ...prev,
          [id]: [
            {
              role: "system",
              content: `❌ **Analysis Failed:** ${errorMessage}`,
            },
          ],
        }));
      } finally {
        setIsUploading(false);
      }
    },
    [],
  );

  const handleNewChat = useCallback(() => {
    setActiveChatId(null);
  }, []);

  const setMessagesForActive = useCallback(
    (updater: React.SetStateAction<ChatMessage[]>) => {
      if (activeChatId === null) return;
      setMessagesByChatId((prev) => ({
        ...prev,
        [activeChatId]:
          typeof updater === "function"
            ? updater(prev[activeChatId] ?? [])
            : updater,
      }));
    },
    [activeChatId],
  );

  const activeMessages = activeChatId
    ? (messagesByChatId[activeChatId] ?? [])
    : [];
  const activeFileName = activeChatId
    ? fileNameByChatId[activeChatId]
    : undefined;

  return (
    <div className="flex min-h-0 flex-1">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={handleNewChat}
      />
      <main className="flex min-w-0 flex-1 overflow-auto">
        {isUploading ? (
          <div className="w-full max-w-2xl mx-auto py-12">
            <CircularDotsLoader className="mx-auto" />
          </div>
        ) : analysisResult ? (
          <div className="w-full mx-auto px-4">
            <div className="flex gap-6">
              <div className="flex-1 min-w-0">
                <Chatbox
                  messages={activeMessages}
                  setMessages={setMessagesForActive}
                  conversationId={activeChatId}
                  fileName={activeFileName}
                />
              </div>
              <div className="w-105 min-w-0">
                <AnalysisTabs analysisResult={analysisResult} />
              </div>
            </div>
          </div>
        ) : activeChatId === null ? (
          <div className="w-full max-w-2xl mx-auto">
            <FileUpload onUpload={handleUpload} />
          </div>
        ) : (
          <div className="w-full mx-auto px-4">
            <div className="flex gap-6">
              <div className="flex-1 min-w-0">
                <Chatbox
                  messages={activeMessages}
                  setMessages={setMessagesForActive}
                  conversationId={activeChatId}
                  fileName={activeFileName}
                />
              </div>
              <div className="w-105 min-w-0">
                <p className="text-center text-gray-500 py-8 bg-white shadow-md rounded-lg p-4">
                  No analysis results yet. Upload a file to begin analysis.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
