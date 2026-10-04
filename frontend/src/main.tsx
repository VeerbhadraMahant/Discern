import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Shell } from "./App";
import { DiscernProvider } from "./state/DiscernContext";
import "./index.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");

createRoot(root).render(
  <StrictMode>
    <DiscernProvider>
      <Shell />
    </DiscernProvider>
  </StrictMode>,
);
