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

import { getSession, setSession, updateSessionRole } from '../../src/sess/session-manager'

beforeEach(() => { localStorage.clear(); setSession('', '', -1, 0) })

test('malformed saved data recovers to an unauthenticated session', () => {
  localStorage.setItem('session', '{broken')
  expect(getSession().status).toBe('unauthenticated')
  expect(localStorage.getItem('session')).toBeNull()
})

test('saved data missing identity fields cannot break the UI', () => {
  localStorage.setItem('session', JSON.stringify({ user: { role: 4 } }))
  expect(getSession().user.role).toBe(0)
})

test('role updates use the latest stored identity and retain expiration state', () => {
  setSession('owner@example.test', 'token-a', 3, -1, 'OrgA', true)
  localStorage.setItem('session', JSON.stringify({ user: { email: 'other@example.test', token: 'token-b', id: 5, role: 1, org: 'OrgB' }, expired: false }))
  updateSessionRole(2)
  expect(getSession().user).toEqual({ email: 'other@example.test', token: 'token-b', id: 5, role: 2, org: 'OrgB' })
  expect(getSession().expired).toBe(false)
})

test('session remains usable when browser storage is blocked', () => {
  const descriptor = Object.getOwnPropertyDescriptor(window, 'localStorage')!
  Object.defineProperty(window, 'localStorage', { configurable: true, get: () => { throw new DOMException('Blocked', 'SecurityError') } })
  try {
    setSession('owner@example.test', 'token', 3, 3, 'OrgA')
    expect(getSession().user.id).toBe(3)
  } finally {
    Object.defineProperty(window, 'localStorage', descriptor)
  }
})
