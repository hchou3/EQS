export default function Header() {
  return (
    <header className="border-b bg-[var(--surface)] px-6 py-4 shadow-sm"
      style={{ borderBottomWidth: "1px", borderColor: "var(--outline)" }}
    >
      <h1 className="text-xl font-semibold tracking-tight text-[var(--foreground)]">
        Clinical AI Platform Equisense
      </h1>
    </header>
  );
}
