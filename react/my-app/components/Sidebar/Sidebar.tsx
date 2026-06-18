"use client";

export type ChatItem = {
  id: string;
  title: string;
  createdAt: Date;
};

type SidebarProps = {
  chats: ChatItem[];
  activeChatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
};

export default function Sidebar({
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
}: SidebarProps) {
  return (
    <aside
      className="flex w-56 shrink-0 flex-col bg-[var(--surface)]"
      style={{ borderRight: "1px solid var(--outline)" }}
      aria-label="Chats"
    >
      <div
        className="p-3"
        style={{ borderBottom: "1px solid var(--outline)" }}
      >
        <button
          type="button"
          onClick={onNewChat}
          className="w-full rounded-md px-3 py-2 text-left text-sm font-medium text-[var(--outline)] transition-colors hover:bg-[var(--outline)]/10"
          style={{ border: "1px solid var(--outline)" }}
        >
          + New chat
        </button>
      </div>
      <div className="sidebar-chats flex-1 overflow-y-auto p-2">
        <p className="mb-2 px-2 text-xs font-medium uppercase tracking-wider text-[var(--outline-muted)]">
          Chats
        </p>
        {chats.length === 0 && (
          <p className="px-2 text-sm text-[var(--outline-muted)]">
            No chats yet. Upload a file to start.
          </p>
        )}
        <ul className="space-y-1">
          {chats.map((chat) => (
            <li key={chat.id}>
              <button
                type="button"
                onClick={() => onSelectChat(chat.id)}
                className={`w-full rounded-md px-3 py-2 text-left text-sm truncate transition-colors ${
                  activeChatId === chat.id
                    ? "bg-[var(--outline)]/20 text-[var(--foreground)]"
                    : "text-[var(--outline-muted)] hover:bg-[var(--surface-elevated)] hover:text-[var(--outline)]"
                }`}
                style={{
                  border:
                    activeChatId === chat.id
                      ? "1px solid var(--outline)"
                      : "1px solid transparent",
                }}
                title={chat.title}
              >
                {chat.title}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
