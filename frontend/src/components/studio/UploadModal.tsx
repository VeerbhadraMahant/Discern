import { useState, useRef, useCallback } from "react";
import { Upload, X, Terminal } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onUploadFile: (file: File) => void;
  isLoading: boolean;
}

export default function UploadModal({ isOpen, onClose, onUploadFile, isLoading }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((files: FileList | null) => {
    const file = files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm font-mono">
      <div className="relative w-full max-w-lg border-2 border-hermes-blue bg-white p-6 text-hermes-ink shadow-terminal">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-hermes-muted hover:text-hermes-ink"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="space-y-1.5 pb-3 border-b border-hermes-ink/15">
          <div className="flex items-center gap-2 text-xs">
            <span className="flex h-2 w-2 bg-hermes-blue" />
            <span className="uppercase text-hermes-muted font-bold">
              // STREAM_FRAME_INGEST
            </span>
          </div>
          <h3 className="hermes-title text-xl font-bold uppercase text-hermes-ink">
            Ingest Camera Frame
          </h3>
          <p className="text-xs text-hermes-charcoal font-body">
            Upload raw CCTV capture (JPG, PNG, WEBP). Discern will classify atmosphere, dispatch OpenCV restoration, and reason about violations.
          </p>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`mt-4 flex flex-col items-center justify-center gap-3 border-2 border-dashed p-6 text-center cursor-pointer transition-all ${
            isDragging
              ? "border-hermes-blue bg-hermes-blue/10"
              : "border-hermes-ink/25 bg-hermes-paper hover:border-hermes-blue"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />

          {previewUrl ? (
            <div className="space-y-2 w-full">
              <div className="aspect-[16/9] w-full overflow-hidden border border-hermes-ink bg-black">
                <img
                  src={previewUrl}
                  alt="Upload Preview"
                  className="h-full w-full object-contain"
                />
              </div>
              <p className="text-xs font-bold text-hermes-ink truncate">
                {selectedFile?.name} ({(selectedFile?.size! / 1024).toFixed(1)} KB)
              </p>
            </div>
          ) : (
            <>
              <div className="flex h-10 w-10 items-center justify-center bg-hermes-blue text-white">
                <Upload className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs font-bold uppercase text-hermes-ink">
                  Drag and drop frame here
                </p>
                <p className="text-[11px] text-hermes-muted mt-0.5">
                  or click to browse files
                </p>
              </div>
            </>
          )}
        </div>

        {/* Actions */}
        <div className="mt-6 flex items-center justify-end gap-3 text-xs">
          <button
            onClick={onClose}
            className="px-4 py-2 uppercase font-bold text-hermes-charcoal hover:text-hermes-ink"
          >
            Cancel
          </button>
          <button
            disabled={!selectedFile || isLoading}
            onClick={() => {
              if (selectedFile) {
                onUploadFile(selectedFile);
                onClose();
              }
            }}
            className="hermes-btn-primary bg-hermes-blue text-white hover:bg-hermes-dark disabled:opacity-40"
          >
            <span>{isLoading ? "ANALYZING..." : "EXECUTE_AGENT"}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
