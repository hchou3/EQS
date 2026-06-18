"use client";

import { useCallback, useState } from "react";

type FileUploadProps = {
  onUpload: (file: File) => void;
  accept?: string;
  maxSizeBytes?: number;
};

export default function FileUpload({
  onUpload,
  accept = ".csv,.json,.txt,.pdf",
  maxSizeBytes = 50 * 1024 * 1024,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndEmit = useCallback(
    (file: File) => {
      setError(null);
      if (file.size > maxSizeBytes) {
        setError(`File must be under ${Math.round(maxSizeBytes / 1024 / 1024)}MB`);
        return;
      }
      onUpload(file);
    },
    [onUpload, maxSizeBytes]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) validateAndEmit(file);
    },
    [validateAndEmit]
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
    [validateAndEmit]
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
      {error && (
        <p className="text-sm text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
