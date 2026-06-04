import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

export function LoginPage({ onLogin }: { onLogin: (token: string, role: string, name: string) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: username, password }),
      });
      if (!res.ok) {
        setError("Неверный логин или пароль");
        return;
      }
      const data = await res.json();
      onLogin(data.token, data.role, data.name);
    } catch {
      setError("Ошибка подключения к серверу");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <Card className="w-full max-w-sm shadow-lg">
        <CardHeader className="text-center">
          <div className="mx-auto mb-3 w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
            <span className="text-xl font-bold text-white">КК</span>
          </div>
          <CardTitle className="text-xl">Карта Компетенций</CardTitle>
          <CardDescription>Войдите в систему</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Логин</Label>
              <Input id="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="admin@... / teacher@... / student@..." required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Пароль</Label>
              <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••" required />
            </div>
            {error && <p className="text-sm text-red-600">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Вход..." : "Войти"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
