// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

export interface Session {
  user: { email: string; token: string; id: number; role: number; org?: string }
  expired?: boolean
  status: string
}

const anonymous = (): Session => ({ user: { email: '', token: '', id: -1, role: 0 }, status: 'unauthenticated' })
let session = anonymous()
let sessionWriteFailed = false

export function setSession(email: string, token: string, id: number, role: number, org?: string, expired = false): Session {
  session = { user: { email, token, id, role, org }, expired, status: role === 0 ? 'unauthenticated' : 'authenticated' }
  try {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('session', JSON.stringify(session))
      sessionWriteFailed = false
    }
  } catch {
    sessionWriteFailed = true
  }
  return session
}

export function getSession(): Session {
  if (typeof window === 'undefined' || sessionWriteFailed) return session
  try {
    const stored = window.localStorage.getItem('session')
    if (!stored) return session = anonymous()
    const candidate = JSON.parse(stored) as Partial<Session>
    const user = candidate?.user
    if (!user || typeof user.email !== 'string' || typeof user.token !== 'string' ||
        !Number.isFinite(user.id) || !Number.isFinite(user.role)) {
      window.localStorage.removeItem('session')
      return session = anonymous()
    }
    session = { user, expired: Boolean(candidate.expired), status: user.role === 0 ? 'unauthenticated' : 'authenticated' }
  } catch (error) {
    if (error instanceof SyntaxError) {
      session = anonymous()
      try { window.localStorage.removeItem('session') } catch { /* Storage may also be unavailable. */ }
    }
  }
  return session
}

export function updateSessionRole(role: number): Session {
  const current = getSession()
  return setSession(current.user.email, current.user.token, current.user.id, role, current.user.org, current.expired)
}

export function printSession(): void {
  const current = getSession()
  console.log({ id: current.user.id, role: current.user.role, status: current.status, expired: current.expired })
}
