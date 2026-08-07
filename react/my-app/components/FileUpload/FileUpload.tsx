"use client";

import { useState, useCallback } from "react";

type FileUploadProps = {
  onUpload: (file: File, llm: string, apiKey?: string) => void;
  accept?: string;
  maxSizeBytes?: number;
};

export default function FileUpload({
  onUpload,
  accept = ".csv",
  maxSizeBytes = 50 * 1024 * 1024,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLlm, setSelectedLlm] = useState<string>("gemini-2.0-flash");
  const [apiKey, setApiKey] = useState<string>("");
  const [showApiModal, setShowApiModal] = useState(false);

  const availableLLMs = [
    { value: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
    { value: "gemini-2.0-pro", label: "Gemini 2.0 Pro" },
    { value: "openai/gpt-4", label: "OpenAI GPT-4" },
    { value: "groq/llama3", label: "Groq Llama3" },
  ];

  const requiresApiKey = selectedLlm !== "gemini-2.0-flash";

  const handleLlmChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setSelectedLlm(e.target.value);
      if (e.target.value !== "gemini-2.0-flash" && !apiKey) {
        setShowApiModal(true);
      }
    },
    [apiKey],
  );

  const handleApiKeyChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setApiKey(e.target.value);
      if (showApiModal) setShowApiModal(false);
    },
    [showApiModal],
  );

  const handleModalSubmit = useCallback(() => {
    if (apiKey.trim()) {
      setShowApiModal(false);
    }
  }, [apiKey]);

  const validateAndEmit = useCallback(
    (file: File) => {
      setError(null);
      if (file.size > maxSizeBytes) {
        setError(
          `File must be under ${Math.round(maxSizeBytes / 1024 / 1024)}MB`,
        );
        return;
      }
      // Pass llm and apiKey to the upload handler
      onUpload(file, selectedLlm, requiresApiKey ? apiKey : undefined);
    },
    [onUpload, maxSizeBytes, selectedLlm, requiresApiKey, apiKey],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) validateAndEmit(file);
    },
    [validateAndEmit],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndEmit(file);
      e.target.value = "";
    },
    [validateAndEmit],
  );

  return (
    <div className="flex flex-col items-center justify-center gap-4 p-8">
      <p className="text-sm text-[var(--outline-muted)]">
        Upload a file to start a new chat
      </p>
      <label
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`flex min-h-[200px] w-full max-w-md cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 transition-colors ${
          isDragging
            ? "border-[var(--outline)] bg-[var(--outline)]/10"
            : "border-[var(--outline-muted)] bg-[var(--surface)] hover:border-[var(--outline)] hover:bg-[var(--outline)]/5"
        }`}
      >
        <input
          type="file"
          accept={accept}
          onChange={handleChange}
          className="hidden"
          aria-label="Upload file"
        />
        <span className="text-[var(--outline)] mb-1 text-lg font-medium">
          Drop a file here or click to browse
        </span>
        <span className="text-center text-sm text-[var(--outline-muted)]">
          {accept}
        </span>
      </label>

      {/* LLM Selection Dropdown */}
      <div className="w-full max-w-md">
        <label
          htmlFor="llm-select"
          className="block text-sm font-medium text-[var(--outline)] mb-1"
        >
          Select LLM Provider
        </label>
        <select
          id="llm-select"
          value={selectedLlm}
          onChange={handleLlmChange}
          className="w-full px-3 py-2 border border-[var(--outline-muted)] rounded-lg bg-[var(--surface)] text-[var(--text)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
        >
          {availableLLMs.map((llm) => (
            <option key={llm.value} value={llm.value}>
              {llm.label}
            </option>
          ))}
        </select>
      </div>

      {/* API Key Input */}
      {requiresApiKey && (
        <div className="w-full max-w-md">
          <label
            htmlFor="api-key-input"
            className="block text-sm font-medium text-[var(--outline)] mb-1"
          >
            API Key
          </label>
          <input
            id="api-key-input"
            type="password"
            placeholder="Enter your API key here"
            value={apiKey}
            onChange={handleApiKeyChange}
            className="w-full px-3 py-2 border border-[var(--outline-muted)] rounded-lg bg-[var(--surface)] text-[var(--text)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
          />
        </div>
      )}

      {/* API Key Modal */}
      {requiresApiKey && !apiKey && showApiModal && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowApiModal(false)}
        >
          <div
            className="bg-[var(--surface)] rounded-lg p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-medium mb-4">API Key Required</h3>
            <p className="text-sm text-[var(--outline-muted)] mb-4">
              The selected LLM provider requires an API key. Please enter it
              below.
            </p>
            <input
              type="password"
              placeholder="Enter your API key here"
              value={apiKey}
              onChange={handleApiKeyChange}
              className="w-full px-3 py-2 border border-[var(--outline-muted)] rounded-lg bg-[var(--surface)] text-[var(--text)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] mb-4"
            />
            <button
              onClick={handleModalSubmit}
              className="w-full px-4 py-2 bg-[var(--primary)] text-white rounded-lg hover:opacity-90 transition-opacity"
            >
              Submit API Key
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="text-sm text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
