import { act, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { backendError } from "../api/errors";
import { MockDiscernClient } from "../api/mock";
import type { AskResponse } from "../api/types";
import { renderApp } from "../test/helpers";
import { useDiscern } from "./DiscernContext";

function Probe() {
  const d = useDiscern();
  return (
    <div>
      <p data-testid="chat">{d.chat.length}</p>
      <p data-testid="session">{d.session ? d.session.id : "none"}</p>
      <p data-testid="error">{d.error ? d.error.kind : "none"}</p>
      <button onClick={() => void d.ask("how many cars")}>ask</button>
      <button onClick={() => void d.startOver()}>over</button>
    </div>
  );
}

describe("session state", () => {
  it("drops the reply to a question that was still running when the visitor started over", async () => {
    let release: (r: AskResponse) => void = () => undefined;
    class Slow extends MockDiscernClient {
      override ask(sid: string, q: string): Promise<AskResponse> {
        const real = super.ask(sid, q);
        return new Promise((resolve) => {
          release = () => void real.then(resolve);
        });
      }
    }
    renderApp(<Probe />, { client: new Slow({ delayMs: 0 }) });
    act(() => screen.getByText("ask").click());
    await waitFor(() => expect(screen.getByTestId("chat")).toHaveTextContent("1"));
    await act(async () => screen.getByText("over").click());
    expect(screen.getByTestId("chat")).toHaveTextContent("0");
    await act(async () => {
      release({} as AskResponse);
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(screen.getByTestId("chat")).toHaveTextContent("0");
    expect(screen.getByTestId("session")).toHaveTextContent("none");
  });

  it("forgets the session when the app says it expired", async () => {
    class Expired extends MockDiscernClient {
      override async ask(): Promise<AskResponse> {
        throw backendError("this session is unknown or has expired; please upload again");
      }
    }
    renderApp(<Probe />, { client: new Expired({ delayMs: 0 }) });
    await act(async () => screen.getByText("ask").click());
    await waitFor(() => expect(screen.getByTestId("error")).toHaveTextContent("expired"));
    expect(screen.getByTestId("session")).toHaveTextContent("none");
    expect(sessionStorage.getItem("discern.session")).toBeNull();
  });
});
