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
  const [isUploading, setIsUploading] = useState(false);

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

      try {
        const formData = new FormData();
        formData.append("file", file);

        // Normalize the LLM value to provider name (e.g., "groq/llama3" -> "groq")
        const provider = llm.includes("/") ? llm.split("/")[0] : llm;
        if (provider) formData.append("llm_provider", provider);

        // DEBUG: Log what we're actually sending
        console.log("Original llm:", llm, "Normalized provider:", provider, "Has apiKey:", !!apiKey);

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

        // Add analysis results as system messages
        const analysisMessages: ChatMessage[] = [
          {
            role: "system",
            content: `📊 **Analysis Complete**\n\n**Equity Score:** ${data.equity_score?.toFixed(2) || "N/A"}\n\n**Dataset:** ${data.dataset_info?.n_rows} rows × ${data.dataset_info?.n_cols} columns`,
          },
        ];

        // Add disparity summaries
        if (data.disparity_summary && data.disparity_summary.length > 0) {
          data.disparity_summary.forEach((d: any) => {
            analysisMessages.push({
              role: "system",
              content: `⚖️ **Bias Check: ${d.protected_attr} → ${d.target_col}**\n\n- Task: ${d.task_type}\n- Disparity: ${d.disparity?.toFixed(4) || "N/A"}\n- P-value: ${d.p_value?.toFixed(4) || "N/A"}\n${d.error ? `- Error: ${d.error}` : ""}`,
            });
          });
        }

        // Add LLM classification info
        if (data.llm_classifications) {
          analysisMessages.push({
            role: "system",
            content: `🤖 **LLM Column Classification**\n\n- Protected Attributes: ${data.llm_classifications.protected_attributes?.join(", ") || "none"}\n- Target Columns: ${data.llm_classifications.target_columns?.join(", ") || "none"}\n- Target Types: ${
              Object.entries(data.llm_classifications.target_column_types || {})
                .map(([k, v]) => `${k}: ${v}`)
                .join(", ") || "N/A"
            }`,
          });
        }

        // Add warnings if any
        if (data.warnings && data.warnings.length > 0) {
          data.warnings.forEach((w: string) => {
            analysisMessages.push({
              role: "system",
              content: `⚠️ **Warning:** ${w}`,
            });
          });
        }

        setMessagesByChatId((prev) => ({ ...prev, [id]: analysisMessages }));
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
