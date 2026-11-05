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

import { Buffer } from 'node:buffer'
import { test, expect, type Page } from '@playwright/test'
import type { ClientFromSchema, Project, User } from '../../src/types'

const NS = '/nvflare-dashboard'
const participantEmail = 'participant@example.test'
const admin: User = { id: 1, name: 'Project Admin', email: 'admin@example.test', role: 'project_admin', organization: 'Project_org', approval_state: 200, created_at: '2026-01-01' }

interface MockDashboard {
  project: Project
  participant: User
  clients: ClientFromSchema[]
  approvals: Record<string, unknown>[]
  roleError?: number
  siteError?: string
  rejectDownload?: boolean
  errors: string[]
}

function dashboard(frozen: boolean): MockDashboard {
  return {
    project: { short_name: 'workshop', title: 'Workshop', description: '', app_location: '', server1: 'server.test', frozen, public: true, num_users: 2, num_clients: 0, num_orgs: 2 },
    participant: { id: 2, name: 'Participant', email: participantEmail, organization: '', role: '', approval_state: 0, created_at: '2026-01-01' },
    clients: [], approvals: [], errors: [],
  }
}

async function routeDashboard(page: Page, state: MockDashboard) {
  page.on('pageerror', error => state.errors.push(error.message))
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.split('/api/v1/')[1]
    const method = request.method()
    const data = method === 'GET' || method === 'DELETE' ? {} : request.postDataJSON()
    const json = (body: Record<string, unknown>, status = 200) => route.fulfill({ status, json: { status: 'ok', ...body } })
    if (path === 'project') {
      if (method === 'PATCH') Object.assign(state.project, data)
      return json({ project: state.project })
    }
    if (path === 'login') return json({ user: data.email === admin.email ? admin : state.participant, access_token: 'test-token' })
    if (path === 'users' && method === 'POST') {
      Object.assign(state.participant, { name: data.name, email: data.email })
      return json({ user: state.participant }, 201)
    }
    if (path === 'users' && method === 'GET') return json({ user_list: [state.participant] })
    if (path === 'users/1') return json({ user: admin })
    if (path === 'users/2') {
      if (method === 'PATCH') {
        if (state.roleError !== undefined && data.role) {
          return json({ status: 'error', message: 'Review the requested role with the Project Admin.' }, state.roleError)
        }
        if (data.approval_state === 100) {
          state.approvals.push(data)
          if (data.reviewed_role !== state.participant.role || data.reviewed_organization !== state.participant.organization) {
            return json({ status: 'error', message: 'Reload and review the registration.' }, 409)
          }
        }
        for (const key of ['role', 'organization', 'approval_state', 'name'] as const) {
          if (data[key] !== undefined) Object.assign(state.participant, { [key]: data[key] })
        }
      }
      return json({ user: state.participant })
    }
    if (path === 'clients' && method === 'GET') return json({ client_list: state.clients })
    if (path === 'clients' && method === 'POST') {
      if ('props' in data) return json({ status: 'error', message: 'Client properties are not accepted.' }, 400)
      const client: ClientFromSchema = { ...data, id: 3, approval_state: 0 }
      state.clients.push(client)
      return json({ client }, 201)
    }
    if (path === 'clients/3' && method === 'PATCH') {
      if (state.siteError) return json({ status: 'error', message: state.siteError }, 403)
      Object.assign(state.clients[0], data)
      return json({ client: state.clients[0] })
    }
    if (path === 'clients/3' && method === 'DELETE') {
      state.clients = []
      return json({})
    }
    if (path.endsWith('/blob')) {
      if (state.rejectDownload) return json({ status: 'error', message: 'The project trust material is not ready. Please retry.' }, 409)
      if (!state.project.frozen || Number(state.participant.approval_state) < 100 || (path.startsWith('clients/') && Number(state.clients[0]?.approval_state ?? 0) < 100)) {
        return json({ status: 'error', message: 'Approval and a frozen project are required.' }, 403)
      }
      return route.fulfill({ status: 200, contentType: 'application/zip', headers: { 'content-disposition': 'attachment; filename="startup.zip"' }, body: Buffer.from('Mock ZIP download; real encrypted-kit coverage runs against Flask.') })
    }
    throw new Error(`Unexpected request: ${method} ${path}`)
  })
}

async function signUp(page: Page) {
  await page.goto(NS + '/registration-form')
  await page.getByLabel('Name', { exact: true }).fill('Participant')
  await page.getByLabel('Email', { exact: true }).fill(participantEmail)
  await page.getByLabel('Password', { exact: true }).fill('Testing1234!')
  await page.getByLabel('Confirm password', { exact: true }).fill('Testing1234!')
  await page.getByRole('button', { name: 'Register', exact: true }).click()
  await page.getByLabel('Organization', { exact: true }).fill('Workshop_org')
  await page.getByRole('combobox', { name: 'Role', exact: true }).click()
  await page.getByRole('option', { name: 'Org Admin', exact: true }).click()
}

for (const frozenAtRegistration of [false, true]) {
  test(`Org Admin signup, approvals and downloads (${frozenAtRegistration ? 'frozen' : 'unfrozen'} project)`, async ({ page, browser, baseURL }) => {
    await page.setViewportSize({ width: 1800, height: 1100 })
    const state = dashboard(frozenAtRegistration)
    await routeDashboard(page, state)
    await signUp(page)
    await page.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(page.getByRole('alert').filter({ hasText: 'Org Admin request pending approval: register your own client sites here.' })).toBeVisible()
    await page.getByRole('button', { name: 'Add Site', exact: true }).click()
    const editor = page.getByRole('dialog')
    await editor.getByLabel('Site Name', { exact: true }).fill('workshop-site')
    await editor.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(editor).not.toBeVisible()
    expect(state.clients[0]).toMatchObject({ name: 'workshop-site', organization: 'Workshop_org', approval_state: 0 })
    await page.getByRole('button', { name: 'Next', exact: true }).click()
    await page.getByRole('button', { name: 'Go to my dashboard', exact: true }).click()
    await expect(page.getByText('Requested role: Org Admin', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'workshop-site', exact: true })).toBeVisible()
    await page.goto(NS + '/downloads')
    await expect(page.getByRole('button', { name: 'Download FLARE Console', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Download Startup Kit', exact: true })).toBeDisabled()

    const adminContext = await browser.newContext({ baseURL, viewport: { width: 1800, height: 1100 } })
    try {
      await adminContext.addInitScript(user => localStorage.setItem('session', JSON.stringify({ user: { ...user, token: 'admin-token', role: 4 }, status: 'authenticated' })), admin)
      const adminPage = await adminContext.newPage()
      await routeDashboard(adminPage, state)
      await adminPage.goto(NS + '/project-admin-dashboard')
      const userRow = adminPage.getByRole('row').filter({ hasText: participantEmail })
      await expect(userRow).toContainText('Org Admin (requested)')
      await userRow.getByRole('checkbox').click()
      await expect(userRow.getByRole('checkbox')).toBeChecked()
      await adminPage.getByRole('button', { name: 'Approve Selected', exact: true }).click()
      await expect(userRow).toContainText('Approved (D/L:')
      expect(state.approvals).toEqual([{ approval_state: 100, reviewed_role: 'org_admin', reviewed_organization: 'Workshop_org' }])
      await adminPage.goto(NS + '/site-dashboard')
      const siteRow = adminPage.getByRole('row').filter({ hasText: 'workshop-site' })
      await siteRow.getByRole('checkbox').click()
      await expect(siteRow.getByRole('checkbox')).toBeChecked()
      await adminPage.getByRole('button', { name: 'Approve Selected', exact: true }).click()
      await expect(siteRow).toContainText('Approved')
      if (!frozenAtRegistration) {
        await adminPage.goto(NS + '/')
        await adminPage.getByRole('button', { name: 'Freeze Project', exact: true }).click()
        await adminPage.getByRole('dialog').getByRole('button', { name: 'Freeze', exact: true }).click()
        await expect.poll(() => state.project.frozen).toBe(true)
      }
      await page.reload()
      const clientDownload = page.getByRole('button', { name: 'Download Startup Kit', exact: true })
      await expect(clientDownload).toBeEnabled()
      state.rejectDownload = true
      await clientDownload.click()
      await expect(page.getByRole('alert').filter({ hasText: 'The project trust material is not ready. Please retry.' })).toBeVisible()
      await expect(page.getByRole('dialog')).not.toBeVisible()
      state.rejectDownload = false
      const downloaded = page.waitForEvent('download')
      await clientDownload.click()
      expect((await downloaded).suggestedFilename()).toBe('startup.zip')
      await expect(page.getByRole('dialog')).toContainText(/The secure PIN is:\s*[A-F0-9]{12}/)
      await page.getByRole('button', { name: 'Okay', exact: true }).click()
      const consoleDownloaded = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Download FLARE Console', exact: true }).click()
      expect((await consoleDownloaded).suggestedFilename()).toBe('startup.zip')
      expect(state.errors).toEqual([])
    } finally {
      await adminContext.close()
    }
  })
}

for (const status of [200, 403, 409, 401]) {
  test(`registration rejection ${status} preserves the role request`, async ({ page }) => {
    const state = dashboard(false)
    state.roleError = status
    await routeDashboard(page, state)
    await signUp(page)
    await page.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(page.getByRole('alert').filter({ hasText: status === 401 ? 'Your session has expired. Please log in again.' : 'Review the requested role with the Project Admin.' })).toBeVisible()
    await expect(page.getByLabel('Organization', { exact: true })).toHaveValue('Workshop_org')
    await expect(page.getByLabel('Organization', { exact: true })).toBeEnabled()
    expect(state.participant.role).toBe('')
    expect(state.errors).toEqual([])
  })
}

test('a stale site-policy error preserves the edited site and session', async ({ page }) => {
  const state = dashboard(false)
  state.participant.organization = 'Workshop_org'
  state.participant.role = 'org_admin'
  state.clients.push({ id: 3, name: 'pending-site', organization: 'Workshop_org', approval_state: 0, capacity: { num_of_gpus: 0, mem_per_gpu_in_GiB: 0 } })
  await page.addInitScript(user => localStorage.setItem('session', JSON.stringify({ user: { ...user, token: 'token', role: 3 }, status: 'authenticated' })), state.participant)
  await routeDashboard(page, state)
  await page.goto(NS + '/user-dashboard')
  await page.getByRole('button', { name: 'pending-site', exact: true }).click()
  const editor = page.getByRole('dialog')
  await editor.getByLabel('Site Name', { exact: true }).fill('unreviewed-site')
  state.siteError = 'Pending applicants can manage only their pending client sites.'
  await editor.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(editor.getByRole('alert').filter({ hasText: state.siteError })).toBeVisible()
  await expect(editor.getByLabel('Site Name', { exact: true })).toHaveValue('unreviewed-site')
  expect(state.clients[0].name).toBe('pending-site')
  expect(page.url()).toContain('/user-dashboard')
  expect(state.errors).toEqual([])
})
