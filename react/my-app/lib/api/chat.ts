/**
 * Sample chat API module.
 * Replace these stubs with real API calls to your backend (e.g. api/api.py).
 */

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
};

export type SendMessageResponse = {
  message: ChatMessage;
  success: boolean;
};

/**
 * API_CALL: Send user message to backend and get assistant reply.
 * Replace with: POST /api/chat or your backend endpoint.
 */
export async function sendMessage(
  message: string,
  _conversationId?: string
): Promise<SendMessageResponse> {
  // TODO: Replace with actual API call, e.g.:
  // const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
  //   method: "POST",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify({ message, conversationId: _conversationId }),
  // });
  // return res.json();

  // Stub response for structure/setup
  return {
    success: true,
    message: {
      id: crypto.randomUUID(),
      role: "assistant",
      content: `[Stub] Echo: ${message}`,
      timestamp: new Date().toISOString(),
    },
  };
}

/**
 * API_CALL: (Optional) Load conversation history on mount or when opening a thread.
 * Replace with: GET /api/chat/history?conversationId=...
 */
export async function getConversationHistory(
  _conversationId: string
): Promise<ChatMessage[]> {
  // TODO: Replace with actual API call.
  return [];
}
