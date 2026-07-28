'use client';

import { useState } from 'react';
import { ArrowRightIcon } from '@phosphor-icons/react';
import { Button } from '@/components/livekit/button';
import { useAuth } from '@/hooks/useAuth';
import { cn } from '@/lib/utils';

const INPUT_CLASS =
  'border-input bg-background text-foreground placeholder:text-muted-foreground ' +
  'flex h-11 w-full rounded-full border px-4 text-sm outline-none transition-colors ' +
  'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] ' +
  'disabled:opacity-50';

/**
 * Two-step phone-OTP login:
 *   1. enter phone number, click "Send code" (server fakes the SMS)
 *   2. enter the code (demo code is 111111), click "Log in"
 *
 * On success `useAuth().login` stores the access token and the gate in
 * `App` swaps this view out for the real session UI.
 */
export function LoginView() {
  const { requestCode, login } = useAuth();
  const [phone, setPhone] = useState('');
  const [code, setCode] = useState('');
  const [codeSent, setCodeSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSendCode() {
    setError(null);
    setBusy(true);
    try {
      await requestCode(phone);
      setCodeSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send code');
    } finally {
      setBusy(false);
    }
  }

  async function handleLogin() {
    setError(null);
    setBusy(true);
    try {
      await login(phone, code);
      // On success the auth gate unmounts this view.
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="bg-background flex min-h-svh flex-col items-center justify-center px-6 text-center">
      <div className="flex w-full max-w-sm flex-col items-center gap-6">
        <div className="flex flex-col gap-1">
          <h1 className="text-foreground text-xl font-semibold">Sign in</h1>
          <p className="text-muted-foreground text-sm">
            Enter your phone number to receive a verification code.
          </p>
        </div>

        <div className="flex w-full flex-col gap-3">
          <input
            className={INPUT_CLASS}
            type="tel"
            inputMode="tel"
            autoComplete="tel"
            placeholder="Phone number"
            value={phone}
            disabled={busy}
            onChange={(e) => setPhone(e.target.value)}
          />

          {!codeSent ? (
            <Button
              variant="primary"
              size="lg"
              className="font-mono"
              disabled={busy || phone.trim().length === 0}
              onClick={handleSendCode}
            >
              Send code
              <ArrowRightIcon weight="bold" />
            </Button>
          ) : (
            <>
              <input
                className={INPUT_CLASS}
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                placeholder="Verification code"
                value={code}
                disabled={busy}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && code.trim().length > 0) handleLogin();
                }}
              />
              <Button
                variant="primary"
                size="lg"
                className="font-mono"
                disabled={busy || code.trim().length === 0}
                onClick={handleLogin}
              >
                Log in
                <ArrowRightIcon weight="bold" />
              </Button>
              <button
                type="button"
                className="text-muted-foreground hover:text-foreground text-xs underline underline-offset-4 disabled:opacity-50"
                disabled={busy}
                onClick={() => {
                  setCodeSent(false);
                  setCode('');
                  setError(null);
                }}
              >
                Use a different phone number
              </button>
            </>
          )}
        </div>

        <p
          className={cn(
            'text-destructive min-h-5 text-sm transition-opacity',
            error ? 'opacity-100' : 'opacity-0'
          )}
          role={error ? 'alert' : undefined}
        >
          {error ?? 'placeholder'}
        </p>
      </div>
    </section>
  );
}
