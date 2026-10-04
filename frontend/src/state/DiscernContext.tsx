import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { GradioDiscernClient } from "../api/client";
import { DiscernError, toDiscernError } from "../api/errors";
import { MockDiscernClient } from "../api/mock";
import type {
  AskResponse,
  CleanResponse,
  DetectResponse,
  DiscernClient,
  IngestDone,
  InfoResponse,
  MediaKind,
  TraceEvent,
} from "../api/types";
import { resolveConfig } from "../config";
import type { AppConfig } from "../config";

export interface Session {
  id: string;
  kind: MediaKind;
  name: string;
  sizeBytes: number;
}

export type Phase = "idle" | "uploading" | "cleaning" | "detecting" | "ingesting" | "asking";
export type Connection =
  | { status: "connecting" }
  | { status: "connected"; host: string }
  | { status: "unreachable"; error: DiscernError };

export interface IngestState {
  progress: number;
  message: string;
  step: number;
  done: IngestDone | null;
}

export interface ChatTurn {
  id: number;
  role: "user" | "assistant";
  text: string;
  response?: AskResponse;
}

export interface DiscernState {
  client: DiscernClient | null;
  isMock: boolean;
  badSpace: string | null;
  connection: Connection;
  info: InfoResponse | null;
  session: Session | null;
  phase: Phase;
  error: DiscernError | null;
  clean: CleanResponse | null;
  detect: DetectResponse | null;
  ingest: IngestState | null;
  chat: ChatTurn[];
  events: TraceEvent[];
  toast: string | null;
  retryConnect: () => void;
  uploadFile: (file: File) => Promise<void>;
  runClean: () => Promise<void>;
  runDetect: (targets: string) => Promise<void>;
  runIngest: () => Promise<void>;
  ask: (question: string) => Promise<boolean>;
  refreshTrace: () => Promise<void>;
  startOver: () => Promise<void>;
  clearError: () => void;
}

const Ctx = createContext<DiscernState | null>(null);

const SESSION_KEY = "discern.session";

function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Partial<Session>;
    if (typeof s.id === "string" && (s.kind === "image" || s.kind === "video")) {
      return { id: s.id, kind: s.kind, name: String(s.name ?? ""), sizeBytes: Number(s.sizeBytes ?? 0) };
    }
  } catch {
    /* storage unavailable or corrupt: start fresh */
  }
  return null;
}

function saveSession(s: Session | null): void {
  try {
    if (s) sessionStorage.setItem(SESSION_KEY, JSON.stringify(s));
    else sessionStorage.removeItem(SESSION_KEY);
  } catch {
    /* ignore */
  }
}

interface ProviderProps {
  children: ReactNode;
  /** Inject a client (tests, demos). Otherwise one is created from the config. */
  client?: DiscernClient;
  config?: AppConfig;
  initialSession?: Session | null;
}

export function DiscernProvider({ children, client: injected, config, initialSession }: ProviderProps) {
  const cfg = useMemo(() => config ?? resolveConfig(), [config]);
  const [client, setClient] = useState<DiscernClient | null>(injected ?? null);
  const [connection, setConnection] = useState<Connection>(
    injected ? { status: "connected", host: injected.host } : { status: "connecting" },
  );
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [session, setSession] = useState<Session | null>(() =>
    initialSession !== undefined ? initialSession : loadSession(),
  );
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<DiscernError | null>(null);
  const [clean, setClean] = useState<CleanResponse | null>(null);
  const [detect, setDetect] = useState<DetectResponse | null>(null);
  const [ingest, setIngest] = useState<IngestState | null>(null);
  const [chat, setChat] = useState<ChatTurn[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const turnId = useRef(0);
  /** Bumped whenever the session is dropped, so replies to calls started earlier are ignored. */
  const epoch = useRef(0);
  const toastTimer = useRef<number | undefined>(undefined);

  // Connect (and reconnect on retry).
  useEffect(() => {
    let cancelled = false;
    const connect = async () => {
      setConnection({ status: "connecting" });
      try {
        const c = injected ?? (cfg.mock ? new MockDiscernClient() : await GradioDiscernClient.connect(cfg.url));
        const i = await c.info();
        if (cancelled) return;
        setClient(c);
        setInfo(i);
        setConnection({ status: "connected", host: c.host });
      } catch (e) {
        if (!cancelled) setConnection({ status: "unreachable", error: toDiscernError(e) });
      }
    };
    void connect();
    return () => {
      cancelled = true;
    };
  }, [injected, cfg, attempt]);

  const retryConnect = useCallback(() => setAttempt((n) => n + 1), []);
  const clearError = useCallback(() => setError(null), []);

  const addEvents = useCallback((ev: TraceEvent[]) => setEvents((prev) => [...prev, ...ev]), []);

  const resetResults = useCallback(() => {
    epoch.current += 1;
    setClean(null);
    setDetect(null);
    setIngest(null);
    setChat([]);
    setEvents([]);
  }, []);

  const dropSession = useCallback(() => {
    resetResults();
    setSession(null);
    saveSession(null);
  }, [resetResults]);

  const guard = useCallback(
    async <T,>(p: Phase, fn: () => Promise<T>): Promise<T | undefined> => {
      setPhase(p);
      setError(null);
      try {
        return await fn();
      } catch (e) {
        const err = toDiscernError(e);
        if (err.kind === "expired") dropSession();
        setError(err);
        return undefined;
      } finally {
        setPhase("idle");
      }
    },
    [dropSession],
  );

  const doClean = useCallback(
    async (c: DiscernClient, sid: string) => {
      const mine = epoch.current;
      const res = await guard("cleaning", () => c.clean(sid));
      if (res && mine === epoch.current) {
        setClean(res);
        addEvents(res.events);
      }
    },
    [guard, addEvents],
  );

  const doIngest = useCallback(
    async (c: DiscernClient, sid: string) => {
      const mine = epoch.current;
      setIngest({ progress: 0, message: "Starting", step: 0, done: null });
      await guard("ingesting", async () => {
        for await (const ev of c.ingest(sid)) {
          if (mine !== epoch.current) return;
          if (ev.stage === "progress") {
            setIngest((prev) => ({ progress: ev.progress, message: ev.message, step: (prev?.step ?? 0) + 1, done: null }));
          } else {
            setIngest((prev) => ({ progress: 1, message: "Done", step: prev?.step ?? 0, done: ev }));
            addEvents(ev.events);
          }
        }
      });
    },
    [guard, addEvents],
  );

  const uploadFile = useCallback(
    async (file: File) => {
      if (!client) return;
      if (session) void client.cleanup(session.id).catch(() => undefined);
      resetResults();
      setSession(null);
      saveSession(null);
      const mine = epoch.current;
      const res = await guard("uploading", () => client.upload(file));
      if (!res) return;
      if (mine !== epoch.current) {
        void client.cleanup(res.session_id).catch(() => undefined); // the visitor started over meanwhile
        return;
      }
      const s: Session = { id: res.session_id, kind: res.kind, name: res.name, sizeBytes: res.size_bytes };
      setSession(s);
      saveSession(s);
      if (s.kind === "image") await doClean(client, s.id);
      else await doIngest(client, s.id);
    },
    [client, session, guard, resetResults, doClean, doIngest],
  );

  const runClean = useCallback(async () => {
    if (client && session) await doClean(client, session.id);
  }, [client, session, doClean]);

  const runIngest = useCallback(async () => {
    if (client && session) await doIngest(client, session.id);
  }, [client, session, doIngest]);

  const runDetect = useCallback(
    async (targets: string) => {
      if (!client || !session) return;
      const mine = epoch.current;
      const res = await guard("detecting", () => client.detect(session.id, targets));
      if (res && mine === epoch.current) {
        setDetect(res);
        addEvents(res.events);
      }
    },
    [client, session, guard, addEvents],
  );

  const ask = useCallback(
    async (question: string): Promise<boolean> => {
      if (!client || !session) return false;
      const mine = epoch.current;
      const userId = ++turnId.current;
      setChat((prev) => [...prev, { id: userId, role: "user", text: question }]);
      const res = await guard("asking", () => client.ask(session.id, question));
      if (mine !== epoch.current) return false;
      if (!res) {
        setChat((prev) => prev.filter((t) => t.id !== userId));
        return false;
      }
      setChat((prev) => [...prev, { id: ++turnId.current, role: "assistant", text: res.answer, response: res }]);
      addEvents(res.events);
      return true;
    },
    [client, session, guard, addEvents],
  );

  const refreshTrace = useCallback(async () => {
    if (!client || !session) return;
    const mine = epoch.current;
    const res = await guard("idle", () => client.trace(session.id));
    if (res && mine === epoch.current) setEvents(res.events);
  }, [client, session, guard]);

  const startOver = useCallback(async () => {
    const sid = session?.id;
    resetResults();
    setSession(null);
    setError(null);
    saveSession(null);
    if (client && sid) {
      try {
        await client.cleanup(sid);
        setToast("Your session was deleted.");
      } catch {
        setToast("Could not confirm the deletion. Your session expires on its own.");
      }
      window.clearTimeout(toastTimer.current);
      toastTimer.current = window.setTimeout(() => setToast(null), 4000);
    }
  }, [client, session, resetResults]);

  const value: DiscernState = {
    client,
    isMock: client?.isMock ?? cfg.mock,
    badSpace: cfg.badSpace,
    connection,
    info,
    session,
    phase,
    error,
    clean,
    detect,
    ingest,
    chat,
    events,
    toast,
    retryConnect,
    uploadFile,
    runClean,
    runDetect,
    runIngest,
    ask,
    refreshTrace,
    startOver,
    clearError,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useDiscern(): DiscernState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useDiscern must be used inside DiscernProvider");
  return v;
}
