import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { TeamsPage } from './pages/TeamsPage'
import { TeamDetailPage } from './pages/TeamDetailPage'
import { LineupDetailPage } from './pages/LineupDetailPage'
import { ExplorerPage } from './pages/ExplorerPage'
import { PlayersPage } from './pages/PlayersPage'
import { PlayerDetailPage } from './pages/PlayerDetailPage'
import { ResearchPage } from './pages/ResearchPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="today" element={<Navigate to="/" replace />} />
            <Route path="teams" element={<TeamsPage />} />
            <Route path="teams/:abbr" element={<TeamDetailPage />} />
            <Route path="lineups/:gamePk/:team" element={<LineupDetailPage />} />
            <Route path="explorer" element={<ExplorerPage />} />
            <Route
              path="synergy"
              element={<Navigate to="/research#adjacent-hitter-research" replace />}
            />
            <Route path="players" element={<PlayersPage />} />
            <Route path="players/:id" element={<PlayerDetailPage />} />
            <Route path="research" element={<ResearchPage />} />
            <Route path="compare" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
