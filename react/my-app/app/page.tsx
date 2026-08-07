import Header from "@/components/Header";
import { Dashboard } from "@/components/Dashboard";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-[var(--background)] font-sans">
      <Header />
      <Dashboard />
    </div>
  );
}
