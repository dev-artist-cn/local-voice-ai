'use client';

import { useMemo } from 'react';
import {
  RoomAudioRenderer,
  SessionProvider,
  StartAudio,
  useSession,
} from '@livekit/components-react';
import { SpinnerIcon } from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { LoginView } from '@/components/app/login-view';
import { ViewController } from '@/components/app/view-controller';
import { Button } from '@/components/livekit/button';
import { Toaster } from '@/components/livekit/toaster';
import { useAgentErrors } from '@/hooks/useAgentErrors';
import { AuthProvider, useAuth } from '@/hooks/useAuth';
import { useDebugMode } from '@/hooks/useDebug';
import { getAuthedConnectionDetailsTokenSource, getSandboxTokenSource } from '@/lib/utils';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

function AppSetup() {
  useDebugMode({ enabled: IN_DEVELOPMENT });
  useAgentErrors();

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

/**
 * The real app — only mounted once the user is authenticated. The connection
 * details fetch includes the bearer token via `getAuthedConnectionDetailsTokenSource`.
 */
function AuthedApp({ appConfig }: AppProps) {
  const { phone, logout } = useAuth();
  const tokenSource = useMemo(() => {
    return typeof process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT === 'string'
      ? getSandboxTokenSource(appConfig)
      : getAuthedConnectionDetailsTokenSource();
  }, [appConfig]);

  const session = useSession(
    tokenSource,
    appConfig.agentName ? { agentName: appConfig.agentName } : undefined
  );

  return (
    <SessionProvider session={session}>
      <AppSetup />
      <main className="grid h-svh grid-cols-1 place-content-center">
        <ViewController appConfig={appConfig} />
      </main>
      <StartAudio label="Start Audio" />
      <RoomAudioRenderer />
      <Toaster />
      <div className="fixed top-4 right-4 z-50 flex items-center gap-2">
        {phone && (
          <span className="text-muted-foreground hidden font-mono text-xs sm:inline">{phone}</span>
        )}
        <Button variant="outline" size="sm" onClick={logout}>
          Sign out
        </Button>
      </div>
    </SessionProvider>
  );
}

export function App({ appConfig }: AppProps) {
  return (
    <AuthProvider>
      <AppContent appConfig={appConfig} />
    </AuthProvider>
  );
}

function AppContent({ appConfig }: AppProps) {
  const { status } = useAuth();

  if (status === 'loading') {
    return (
      <main className="text-muted-foreground grid h-svh place-content-center">
        <SpinnerIcon weight="bold" className="size-6 animate-spin" />
      </main>
    );
  }

  if (status === 'unauthed') {
    return <LoginView />;
  }

  return <AuthedApp appConfig={appConfig} />;
}
