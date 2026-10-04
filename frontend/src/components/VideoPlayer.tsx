import { useImperativeHandle, useRef, useState } from "react";
import type { Ref } from "react";
import { Button } from "./ui";

export interface PlayerHandle {
  seek(seconds: number): void;
}

export function VideoPlayer({
  ref,
  originalUrl,
  annotatedUrl,
  canAnnotate,
  loadingAnnotated,
  onRequestAnnotated,
}: {
  ref?: Ref<PlayerHandle>;
  originalUrl: string;
  annotatedUrl: string | null;
  canAnnotate: boolean;
  loadingAnnotated: boolean;
  onRequestAnnotated: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [showAnnotated, setShowAnnotated] = useState(false);
  const [failed, setFailed] = useState(false);

  useImperativeHandle(ref, () => ({
    seek(t: number) {
      const el = video.current;
      if (!el) return;
      el.currentTime = t;
      el.scrollIntoView?.({ block: "nearest" });
    },
  }));

  const src = showAnnotated && annotatedUrl ? annotatedUrl : originalUrl;
  const toggle = () => {
    if (showAnnotated) {
      setShowAnnotated(false);
    } else if (annotatedUrl) {
      setShowAnnotated(true);
    } else {
      onRequestAnnotated();
      setShowAnnotated(true);
    }
    setFailed(false);
  };

  return (
    <div className="flex flex-col gap-3">
      <video
        key={src}
        ref={video}
        src={src}
        controls
        playsInline
        preload="metadata"
        aria-label={showAnnotated ? "Annotated video" : "Original video"}
        onError={() => setFailed(true)}
        className="block w-full bg-ink"
      />
      {failed && <p role="status">This video could not be loaded in the browser.</p>}
      <div className="flex flex-wrap items-center gap-4">
        <p className="font-semibold">{showAnnotated && annotatedUrl ? "Showing: annotated" : "Showing: original"}</p>
        {canAnnotate && (
          <Button variant="link" onClick={toggle} loading={loadingAnnotated} loadingLabel="Preparing annotated video">
            {showAnnotated ? "Show original" : "Show annotated video"}
          </Button>
        )}
      </div>
    </div>
  );
}
