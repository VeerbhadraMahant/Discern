import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MockDiscernClient } from "../api/mock";
import type { DiscernClient } from "../api/types";
import { DiscernProvider } from "../state/DiscernContext";
import type { Session } from "../state/DiscernContext";

export const SESSION: Session = { id: "demo-1", kind: "video", name: "clip.mp4", sizeBytes: 1000 };

export function renderApp(
  ui: ReactElement,
  opts: { client?: DiscernClient; session?: Session | null } = {},
) {
  const client = opts.client ?? new MockDiscernClient({ delayMs: 0 });
  const session = opts.session === undefined ? SESSION : opts.session;
  return { client, ...render(<DiscernProvider client={client} initialSession={session}>{ui}</DiscernProvider>) };
}
