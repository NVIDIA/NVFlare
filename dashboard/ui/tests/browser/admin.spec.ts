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

import { expect, Page, test } from '@playwright/test'
import { ClientFromSchema, User } from '../../src/types'

const NS = '/nvflare-dashboard'
type UserPatch = Partial<User> & { reviewed_role?: string; reviewed_organization?: string }

async function setupAdmin(page: Page) {
  const state = {
    users: [
      { id: 11, name: 'Owner', email: 'owner@example.com', organization: 'Workshop_org', role: 'org_admin', approval_state: 0, created_at: '2026-01-01', download_count: 0 },
      { id: 12, name: 'Member', email: 'member@example.com', organization: 'Other_org', role: 'member', approval_state: 0, created_at: '2026-01-01', download_count: 0 },
    ] as User[],
    clients: [
      { id: 31, name: 'workshop-site', organization: 'Workshop_org', approval_state: 0, capacity: { num_of_gpus: 4, mem_per_gpu_in_GiB: 16 } },
    ] as ClientFromSchema[],
    userPatches: [] as { id: number; data: UserPatch }[],
    clientPatches: [] as { id: number; data: Partial<ClientFromSchema> }[],
    deletedClients: [] as number[],
    pageErrors: [] as string[],
    holdApprovals: false,
  }
  let releaseApprovals = () => {}
  const bothApprovalsReceived = new Promise<void>(resolve => { releaseApprovals = resolve })
  await page.setViewportSize({ width: 1920, height: 1080 })
  page.on('pageerror', failure => state.pageErrors.push(failure.message))
  await page.addInitScript(() => {
    localStorage.setItem('session', JSON.stringify({
      user: { id: 1, email: 'admin@example.com', token: 'test-token', role: 4 },
      status: 'authenticated',
    }))
  })
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.split('/api/v1/')[1]
    const method = request.method()
    const json = (body: object, status = 200) => route.fulfill({ status, json: { status: 'ok', ...body } })
    if (path === 'project' && method === 'GET') return json({ project: { short_name: 'Workshop' } })
    if (path === 'users' && method === 'GET') return json({ user_list: state.users })
    if (path.startsWith('users/') && method === 'PATCH') {
      const id = Number(path.split('/')[1])
      const user = state.users.find(entry => entry.id === id)
      const data = request.postDataJSON() as UserPatch
      state.userPatches.push({ id, data })
      if (state.holdApprovals && data.approval_state === 100) {
        if (state.userPatches.length === 2) releaseApprovals()
        await bothApprovalsReceived
      }
      if (!user) return json({ status: 'error', message: 'User not found.' }, 404)
      if (data.approval_state === 100 && (data.reviewed_role !== user.role || data.reviewed_organization !== user.organization)) {
        return json({ status: 'error', message: 'Review the current role and organization before approving this user.' }, 409)
      }
      Object.assign(user, data)
      return json({ user })
    }
    if (path === 'clients' && method === 'GET') return json({ client_list: state.clients })
    if (path.startsWith('clients/') && method === 'PATCH') {
      const id = Number(path.split('/')[1])
      const data = request.postDataJSON() as Partial<ClientFromSchema>
      state.clientPatches.push({ id, data })
      if (data.name === 'duplicate-name') return json({ status: 'error', message: 'This site name already exists. Please choose another site name.' }, 409)
      const client = state.clients.find(entry => entry.id === id)
      if (!client) return json({ status: 'error', message: 'Client site not found.' }, 404)
      Object.assign(client, data)
      return json({ client })
    }
    if (path.startsWith('clients/') && method === 'DELETE') {
      const id = Number(path.split('/')[1])
      state.deletedClients.push(id)
      state.clients = state.clients.filter(client => client.id !== id)
      return json({})
    }
    throw new Error(`Unexpected request: ${method} ${path}`)
  })
  return state
}

test('real DataGrid bulk approval reviews both identities and retains every successful update', async ({ page }) => {
  const state = await setupAdmin(page)
  state.holdApprovals = true
  await page.goto(NS + '/project-admin-dashboard')
  const owner = page.getByRole('row').filter({ hasText: 'owner@example.com' })
  const member = page.getByRole('row').filter({ hasText: 'member@example.com' })
  await expect(owner).toContainText('Org Admin (requested)')
  await expect(owner).toContainText('Workshop_org')
  await expect(member).toContainText('Other_org')
  await owner.getByRole('checkbox').click()
  await expect(owner.getByRole('checkbox')).toBeChecked()
  await member.getByRole('checkbox').click()
  await expect(member.getByRole('checkbox')).toBeChecked()
  await page.getByRole('button', { name: 'Approve Selected', exact: true }).click()
  await expect(owner).toContainText('Approved (D/L: 0)')
  await expect(member).toContainText('Approved (D/L: 0)')
  await expect(owner).not.toContainText('Org Admin (requested)')
  expect(state.userPatches).toEqual(expect.arrayContaining([
    { id: 11, data: { approval_state: 100, reviewed_role: 'org_admin', reviewed_organization: 'Workshop_org' } },
    { id: 12, data: { approval_state: 100, reviewed_role: 'member', reviewed_organization: 'Other_org' } },
  ]))
  expect(state.pageErrors).toEqual([])
})

test('a stale reviewed registration remains pending with an actionable error', async ({ page }) => {
  const state = await setupAdmin(page)
  await page.goto(NS + '/project-admin-dashboard')
  const owner = page.getByRole('row').filter({ hasText: 'owner@example.com' })
  await expect(owner).toContainText('Workshop_org')
  state.users[0].organization = 'Corrected_org'
  await owner.getByRole('checkbox').click()
  await expect(owner.getByRole('checkbox')).toBeChecked()
  await page.getByRole('button', { name: 'Approve Selected', exact: true }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'Review the current role and organization' })).toBeVisible()
  await expect(owner).toContainText('Pending')
  await expect(owner.getByRole('checkbox')).toBeChecked()
  expect(state.users[0].approval_state).toBe(0)
  expect(state.pageErrors).toEqual([])
})

test('real site controls preserve a rejected rename and support approval, denial, and confirmed deletion', async ({ page }) => {
  const state = await setupAdmin(page)
  await page.goto(NS + '/site-dashboard')
  await page.getByRole('button', { name: 'Edit name for workshop-site', exact: true }).click()
  const editor = page.getByRole('textbox', { name: 'Site Name', exact: true })
  await editor.fill('duplicate-name')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'This site name already exists.' })).toBeVisible()
  await expect(editor).toHaveValue('duplicate-name')
  expect(state.clients[0].name).toBe('workshop-site')
  await editor.fill('corrected-site')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByRole('dialog')).not.toBeVisible()
  const row = page.getByRole('row').filter({ hasText: 'corrected-site' })
  await expect(row).toBeVisible()
  await row.getByRole('checkbox').click()
  await expect(row.getByRole('checkbox')).toBeChecked()
  await page.getByRole('button', { name: 'Approve Selected', exact: true }).click()
  await expect(row).toContainText('Approved')
  await row.getByRole('checkbox').click()
  await expect(row.getByRole('checkbox')).toBeChecked()
  await page.getByRole('button', { name: 'Deny Selected', exact: true }).click()
  await expect(row).toContainText('Denied')
  await row.getByRole('checkbox').click()
  await expect(row.getByRole('checkbox')).toBeChecked()
  await page.getByRole('button', { name: 'Delete Selected', exact: true }).click()
  expect(state.deletedClients).toEqual([])
  await page.getByRole('button', { name: 'Delete Sites', exact: true }).click()
  await expect.poll(() => state.deletedClients).toEqual([31])
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await expect(row).toHaveCount(0)
  expect(state.clientPatches).toEqual(expect.arrayContaining([
    { id: 31, data: { name: 'duplicate-name' } },
    { id: 31, data: { name: 'corrected-site' } },
    { id: 31, data: { approval_state: 100 } },
    { id: 31, data: { approval_state: -1 } },
  ]))
  expect(state.pageErrors).toEqual([])
})
