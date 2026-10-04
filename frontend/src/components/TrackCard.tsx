import { CheckCircle, Prohibit } from "@phosphor-icons/react";
import type { Track } from "../api/types";
import { formatRange } from "../lib/format";
import { Button, Card, Stamp } from "./ui";

export function TrackCard({ track, onJump }: { track: Track; onJump: (seconds: number) => void }) {
  const accepted = track.status === "accepted";
  return (
    <Card as="li" className="flex flex-col gap-3">
      <div className="flex gap-4">
        {track.crop_url ? (
          <img
            src={track.crop_url}
            alt={`Crop of ${track.label}, ${formatRange(track.t_start, track.t_end)}`}
            className="size-24 shrink-0 bg-ink object-cover"
          />
        ) : (
          <div className="flex size-24 shrink-0 items-center justify-center bg-parchment text-center type-body-s">No crop</div>
        )}
        <div className="flex min-w-0 flex-col gap-1">
          <p className="type-h3">
            {track.label} <span className="num ident">#{track.id}</span>
          </p>
          <p className="num font-normal">{formatRange(track.t_start, track.t_end)}</p>
          <p className="num font-normal">
            mean confidence {Math.round(track.mean_score * 100)}%, {track.n_frames} frames
          </p>
          <div>
            <Stamp icon={accepted ? <CheckCircle size={16} aria-hidden="true" /> : <Prohibit size={16} aria-hidden="true" />}>
              {track.status}
            </Stamp>
          </div>
        </div>
      </div>
      <p className="font-normal">{track.rationale}</p>
      <div>
        <Button variant="link" onClick={() => onJump(track.t_start)} aria-label={`Jump to time ${formatRange(track.t_start, track.t_end)} for ${track.label} number ${track.id}`}>
          Jump to time
        </Button>
      </div>
    </Card>
  );
}
