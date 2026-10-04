import { useId, useState } from "react";

export function BeforeAfter({
  beforeUrl,
  afterUrl,
  beforeAlt,
  afterAlt,
}: {
  beforeUrl: string;
  afterUrl: string;
  beforeAlt: string;
  afterAlt: string;
}) {
  const id = useId();
  const [compare, setCompare] = useState(false);
  const [pos, setPos] = useState(50);

  return (
    <div className="flex flex-col gap-3">
      <label className="flex min-h-11 items-center gap-2 font-normal">
        <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} className="size-5 shrink-0 accent-ink" />
        Compare with a slider
      </label>
      {compare ? (
        <div className="flex flex-col gap-3">
          <div className="relative overflow-hidden bg-ink">
            <img src={beforeUrl} alt={beforeAlt} className="block w-full" />
            <img
              src={afterUrl}
              alt={afterAlt}
              className="absolute inset-0 block h-full w-full object-cover"
              style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
            />
            <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-parchment" style={{ left: `${pos}%` }} aria-hidden="true" />
          </div>
          <label htmlFor={id} className="font-semibold">
            Slider position: after on the left, before on the right
          </label>
          <input
            id={id}
            type="range"
            min={0}
            max={100}
            step={1}
            value={pos}
            onChange={(e) => setPos(Number(e.target.value))}
            aria-valuetext={`${pos} percent after`}
            className="min-h-11 w-full accent-ink"
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <figure className="flex flex-col gap-2">
            <img src={beforeUrl} alt={beforeAlt} className="block w-full bg-ink" />
            <figcaption className="type-body-s">Before</figcaption>
          </figure>
          <figure className="flex flex-col gap-2">
            <img src={afterUrl} alt={afterAlt} className="block w-full bg-ink" />
            <figcaption className="type-body-s">After</figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
