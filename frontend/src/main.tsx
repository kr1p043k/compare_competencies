import React from 'react';
import { createRoot } from "react-dom/client";
import App from "./app/App.tsx";
import { AuthProvider } from "./lib/auth.tsx";
import "./styles/index.css";

// Убеждаемся, что React загружен
console.log('React version:', React.version);

const root = document.getElementById("root");
if (!root) {
  throw new Error("Root element not found");
}

createRoot(root).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>
);