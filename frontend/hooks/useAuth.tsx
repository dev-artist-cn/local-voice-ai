'use client';

import {
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { clearSession, getAccessToken, getStoredPhone, setSession } from '@/lib/auth';

export type AuthStatus = 'loading' | 'authed' | 'unauthed';

export interface AuthApi {
  status: AuthStatus;
  isAuthenticated: boolean;
  phone: string | null;
  requestCode: (phone: string) => Promise<void>;
  login: (phone: string, code: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthApi | null>(null);

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return typeof data?.detail === 'string' ? data.detail : '';
  } catch {
    return '';
  }
}

/**
 * Single source of truth for the phone-OTP session, shared across the app via
 * context. Without this, every component calling `useAuth()` would get its own
 * isolated state — so a login inside `LoginView` would never flip `App`'s gate.
 *
 * Hydrates from localStorage on mount (starts in `loading` to avoid SSR/prerender
 * hydration mismatches) and keeps tabs roughly in sync via the `storage` event.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [phone, setPhone] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setToken(getAccessToken());
    setPhone(getStoredPhone());
    setHydrated(true);

    const onStorage = (e: StorageEvent) => {
      if (e.key === 'lva.access_token') {
        setToken(e.newValue);
        if (!e.newValue) setPhone(null);
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const requestCode = useCallback(async (rawPhone: string) => {
    const res = await fetch('/api/auth/request-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone: rawPhone.trim() }),
    });
    if (!res.ok) {
      throw new Error((await readError(res)) || `request-code failed (${res.status})`);
    }
  }, []);

  const login = useCallback(async (rawPhone: string, code: string) => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone: rawPhone.trim(), code: code.trim() }),
    });
    if (!res.ok) {
      throw new Error((await readError(res)) || `login failed (${res.status})`);
    }
    const data = (await res.json()) as { access_token: string; phone?: string };
    setSession(data.access_token, data.phone ?? rawPhone.trim());
    setToken(data.access_token);
    setPhone(data.phone ?? rawPhone.trim());
  }, []);

  const logout = useCallback(() => {
    clearSession();
    setToken(null);
    setPhone(null);
  }, []);

  const status: AuthStatus = !hydrated ? 'loading' : token ? 'authed' : 'unauthed';

  const value = useMemo<AuthApi>(
    () => ({ status, isAuthenticated: status === 'authed', phone, requestCode, login, logout }),
    [status, phone, requestCode, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthApi {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}
