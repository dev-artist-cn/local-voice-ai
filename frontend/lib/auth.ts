/**
 * Browser-side storage for the phone-OTP access token.
 *
 * The token is an HS256 JWT minted by `POST /api/auth/login` and sent back to
 * the server as `Authorization: Bearer <token>` on protected calls (notably
 * `/api/connection-details`). We keep it in localStorage so a page reload
 * preserves the session.
 *
 * This is a demo-grade store. For a real app consider httpOnly cookies +
 * CSRF tokens, refresh-token rotation, etc.
 */

const TOKEN_KEY = 'lva.access_token';
const PHONE_KEY = 'lva.phone';

export function getAccessToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function getStoredPhone(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(PHONE_KEY);
}

export function setSession(accessToken: string, phone: string): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(TOKEN_KEY, accessToken);
  window.localStorage.setItem(PHONE_KEY, phone);
}

export function clearSession(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(PHONE_KEY);
}
