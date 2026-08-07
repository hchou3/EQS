"use client";

import { useState, useCallback, useEffect } from "react";

type FileUploadProps = {
  onUpload: (file: File, llm: string, apiKey?: string) => void;
  accept?: string;
  maxSizeBytes?: number;
};

// Map dropdown values to provider names used by the backend
const LLM_TO_PROVIDER_MAP: Record<string, string> = {
  "gemini-2.0-flash": "gemini",
  "gemini-2.0-pro": "gemini",
  "openai/gpt-4": "openai",
  "groq/llama3": "groq",
};

// Map dropdown values to display labels
const LLM_LABELS: Record<string, string> = {
  "gemini-2.0-flash": "Gemini 2.0 Flash",
  "gemini-2.0-pro": "Gemini 2.0 Pro",
  "openai/gpt-4": "OpenAI GPT-4",
  "groq/llama3": "Groq Llama3",
};

export default function FileUpload({
  onUpload,
  accept = ".csv",
  maxSizeBytes = 50 * 1024 * 1024,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLlm, setSelectedLlm] = useState<string>(""); // Default to empty (no selection)
  const [apiKey, setApiKey] = useState<string>("");
  const [providerConfig, setProviderConfig] = useState<Record<string, boolean>>(
    {},
  );
  const [configLoaded, setConfigLoaded] = useState(false);

  const availableLLMs = [
    { value: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
    { value: "gemini-2.0-pro", label: "Gemini 2.0 Pro" },
    { value: "openai/gpt-4", label: "OpenAI GPT-4" },
    { value: "groq/llama3", label: "Groq Llama3" },
  ];

  // Normalize LLM selection value to provider name
  const getProviderName = (llmValue: string): string => {
    return LLM_TO_PROVIDER_MAP[llmValue] || llmValue;
  };

  // Check if backend has API key configured for a provider
  const backendHasKey = (llmValue: string): boolean => {
    const provider = getProviderName(llmValue);
    if (!configLoaded) return false;
    return Boolean(providerConfig[provider]);
  };

  // Fetch provider config on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/provider-config`,
        );
        if (response.ok) {
          const data = await response.json();
          setProviderConfig(data.providers || {});
        }
      } catch (err) {
        console.error("Failed to load provider config:", err);
      } finally {
        setConfigLoaded(true);
      }
    };
    void fetchConfig();
  }, []);

  const handleLlmChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const newLlm = e.target.value;
      setSelectedLlm(newLlm);
      // Clear any previously entered API key when switching providers
      setApiKey("");
    },
    [],
  );

  const handleApiKeyChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setApiKey(e.target.value);
    },
    [],
  );

  const validateAndEmit = useCallback(
    (file: File) => {
      setError(null);

      // Check if provider is selected
      if (!selectedLlm) {
        setError("Please select an LLM provider before uploading.");
        return;
      }

      // Check if provider needs an API key and we have it
      const providerName = getProviderName(selectedLlm);
      const hasBackendKey = backendHasKey(selectedLlm);
      const requiresUserKey = !hasBackendKey;

      if (requiresUserKey && !apiKey.trim()) {
        setError(
          `API key is required for ${LLM_LABELS[selectedLlm]}. Please enter it above.`,
        );
        return;
      }

      if (file.size > maxSizeBytes) {
        setError(
          `File must be under ${Math.round(maxSizeBytes / 1024 / 1024)}MB`,
        );
        return;
      }

      // Pass provider name and apiKey to the upload handler
      onUpload(
        file,
        providerName,
        requiresUserKey ? apiKey : hasBackendKey ? undefined : apiKey,
      );
    },
    [onUpload, maxSizeBytes, selectedLlm, apiKey, configLoaded, providerConfig],
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

  // Compute provider status for display
  const providerStatus = selectedLlm ? getProviderName(selectedLlm) : null;
  const hasKeyInBackend = providerStatus ? backendHasKey(selectedLlm) : false;
  const needsUserKey = selectedLlm && !hasKeyInBackend;

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
          <option value="" disabled hidden>
            -- Select a provider --
          </option>
          {availableLLMs.map((llm) => (
            <option key={llm.value} value={llm.value}>
              {llm.label}
            </option>
          ))}
        </select>

        {/* Provider Status Message */}
        {selectedLlm && configLoaded && (
          <div className="mt-2 text-sm">
            {hasKeyInBackend ? (
              <p className="text-green-500">
                ✓ Using {LLM_LABELS[selectedLlm]} key from .env
              </p>
            ) : (
              <p className="text-yellow-500">
                No API key found for {LLM_LABELS[selectedLlm]}. Please enter one
                below.
              </p>
            )}
          </div>
        )}
      </div>

      {/* API Key Input - shown only when backend doesn't have the key */}
      {needsUserKey && (
        <div className="w-full max-w-md">
          <label
            htmlFor="api-key-input"
            className="block text-sm font-medium text-[var(--outline)] mb-1"
          >
            API Key for {LLM_LABELS[selectedLlm]}
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

      {error && (
        <p className="text-sm text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
