import { UploadSimple } from "@phosphor-icons/react";
import { useId, useState } from "react";
import type { DragEvent } from "react";
import type { Limits } from "../api/types";
import { formatBytes } from "../lib/format";
import { Field } from "./ui";

/** Default ceiling used only until `info` has told us the real limit. */
export function validateFile(file: File, limits: Pick<Limits, "max_upload_mb"> | null): string | null {
  const isImage = file.type.startsWith("image/");
  const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv)$/i.test(file.name);
  if (!isImage && !isVideo) {
    return `"${file.name}" is not an image or a video. Choose a JPEG, PNG, WebP, MP4 or similar file.`;
  }
  if (limits && file.size > limits.max_upload_mb * 1024 * 1024) {
    return `"${file.name}" is ${formatBytes(file.size)}, over the ${limits.max_upload_mb} MB limit. Choose a smaller file or trim the video.`;
  }
  if (file.size === 0) return `"${file.name}" is empty. Choose a different file.`;
  return null;
}

export function DropZone({
  limits,
  disabled,
  onFile,
}: {
  limits: Limits | null;
  disabled?: boolean;
  onFile: (file: File) => void;
}) {
  const id = useId();
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);

  const accept = (file: File | undefined) => {
    if (!file) return;
    const problem = validateFile(file, limits);
    setError(problem);
    if (!problem) onFile(file);
  };

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setOver(false);
    if (!disabled) accept(e.dataTransfer.files[0]);
  };

  return (
    <Field
      id={id}
      label="Media file"
      error={error}
      helper={
        limits
          ? `Images or videos, up to ${limits.max_upload_mb} MB. Videos up to ${limits.max_video_seconds} seconds.`
          : "Images or videos. Limits load once the app connects."
      }
    >
      <label
        htmlFor={id}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`flex min-h-48 flex-col items-center justify-center gap-3 border border-dashed border-ink bg-bone p-6 text-center ${
          over ? "outline outline-2 outline-offset-2 outline-ink" : ""
        } ${disabled ? "cursor-not-allowed opacity-45" : "cursor-pointer"} has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-ember`}
      >
        <UploadSimple size={24} aria-hidden="true" />
        <span className="max-w-[40ch] type-body-l">Drop an image or video, or choose a file</span>
        <input
          id={id}
          type="file"
          accept="image/*,video/*"
          disabled={disabled}
          aria-describedby={`${id}-help${error ? ` ${id}-err` : ""}`}
          aria-invalid={error ? true : undefined}
          className="sr-only"
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </label>
    </Field>
  );
}
