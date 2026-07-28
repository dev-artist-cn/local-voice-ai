import { type ClassValue, clsx } from 'clsx';
import { TokenSource } from 'livekit-client';
import { twMerge } from 'tailwind-merge';
import type { AppConfig } from '@/app-config';
import { clearSession, getAccessToken } from '@/lib/auth';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Get styles for the app
 * @param appConfig - The app configuration
 * @returns A string of styles
 */
export function getStyles(appConfig: AppConfig) {
  const { accent, accentDark } = appConfig;

  return [
    accent
      ? `:root { --primary: ${accent}; --primary-hover: color-mix(in srgb, ${accent} 80%, #000); }`
      : '',
    accentDark
      ? `.dark { --primary: ${accentDark}; --primary-hover: color-mix(in srgb, ${accentDark} 80%, #000); }`
      : '',
  ]
    .filter(Boolean)
    .join('\n');
}

/**
 * Token source for the self-hosted backend. Calls `/api/connection-details`
 * with the phone-OTP access token as a bearer header. The token is read fresh
 * from localStorage on every fetch, so an expired token triggers a 401 → the
 * session is cleared and the page reloads back to the login view.
 */
export function getAuthedConnectionDetailsTokenSource() {
  return TokenSource.custom(async () => {
    const token = getAccessToken();
    const res = await fetch('/api/connection-details', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({}),
    });
    if (res.status === 401) {
      clearSession();
      if (typeof window !== 'undefined') window.location.reload();
      throw new Error('session expired');
    }
    if (!res.ok) {
      throw new Error(`connection-details failed (${res.status})`);
    }
    return (await res.json()) as {
      serverUrl: string;
      roomName: string;
      participantName: string;
      participantToken: string;
    };
  });
}

/**
 * Get a token source for a sandboxed LiveKit session
 * @param appConfig - The app configuration
 * @returns A token source for a sandboxed LiveKit session
 */
export function getSandboxTokenSource(appConfig: AppConfig) {
  return TokenSource.custom(async () => {
    const url = new URL(process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT!, window.location.origin);
    const sandboxId = appConfig.sandboxId ?? '';
    const roomConfig = appConfig.agentName
      ? {
          agents: [{ agent_name: appConfig.agentName }],
        }
      : undefined;

    try {
      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Sandbox-Id': sandboxId,
        },
        body: JSON.stringify({
          room_config: roomConfig,
        }),
      });
      return await res.json();
    } catch (error) {
      console.error('Error fetching connection details:', error);
      throw new Error('Error fetching connection details!');
    }
  });
}
