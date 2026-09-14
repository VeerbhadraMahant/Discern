import { useCallback, useRef, useState } from "react";
import { UploadIcon, CameraIcon } from "./icons";

interface Props {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

export default function UploadPanel({ onSelect, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file && file.type.startsWith("image/")) {
        onSelect(file);
      }
    },
    [onSelect]
  );

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a camera frame to analyze"
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) inputRef.current?.click();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={`flex flex-col items-center justify-center gap-4 rounded-card-lg border-2 border-dashed p-16 text-center transition-colors cursor-pointer
        ${dragging ? "border-accent bg-sky-tint/40" : "border-slate/30 bg-paper hover:border-accent/50"}
        ${disabled ? "pointer-events-none opacity-50" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-obsidian text-paper">
        <UploadIcon />
      </div>
      <div>
        <p className="font-heading text-lg font-semibold text-ink">Drop a camera frame here</p>
        <p className="mt-1 text-sm text-charcoal">or click to browse — JPG, PNG, WEBP</p>
      </div>
      <div className="flex items-center gap-2 rounded-pill bg-cloud px-4 py-2 text-xs font-medium text-charcoal">
        <CameraIcon width={16} height={16} />
        Simulates a live CCTV frame from a site camera
      </div>
    </div>
  );
}
