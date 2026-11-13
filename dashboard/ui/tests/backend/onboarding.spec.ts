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

import { test, expect, Page } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import path from 'node:path'

const NS = '/nvflare-dashboard'
const API = NS + '/api/v1'

async function login(page: Page) {
  await page.goto(NS + '/')
  await page.getByRole('textbox', { name: /^Email/ }).fill('admin@example.test')
  await page.getByLabel(/^Password/).fill('Testing1234!')
  await page.getByRole('button', { name: 'Log in', exact: true }).click()
  await page.waitForURL('**/project-admin-dashboard')
}

for (const frozenAtRegistration of [false, true]) {
  test(`real signup, approval and encrypted kits (${frozenAtRegistration ? 'frozen' : 'unfrozen'} project)`, async ({ page, browser }, testInfo) => {
    const email = `workshop-${Date.now()}@example.test`
    const siteName = `site-${Date.now()}`
    const failures: string[] = []
    page.on('pageerror', error => failures.push(error.message))
    const adminContext = await browser.newContext({ baseURL: 'http://127.0.0.1:3110', viewport: { width: 1600, height: 1000 } })
    const admin = await adminContext.newPage()
    admin.on('pageerror', error => failures.push(error.message))
    try {
      if (frozenAtRegistration) {
        const credentials = await page.request.post(API + '/login', { data: { email: 'admin@example.test', password: 'Testing1234!' } })
        const token = (await credentials.json()).access_token
        const result = await page.request.patch(API + '/project', { headers: { Authorization: `Bearer ${token}` }, data: { frozen: true } })
        expect(result.ok()).toBeTruthy()
      }
      await page.goto(NS + '/registration-form')
      await page.getByLabel('Name', { exact: true }).fill('Workshop Participant')
      await page.getByLabel('Email', { exact: true }).fill(email)
      await page.getByLabel('Password', { exact: true }).fill('Testing1234!')
      await page.getByLabel('Confirm password', { exact: true }).fill('Testing1234!')
      await page.getByRole('button', { name: 'Register', exact: true }).click()
      await page.getByLabel('Organization', { exact: true }).fill('Workshop_org')
      await page.getByRole('combobox', { name: 'Role' }).click()
      await page.getByRole('option', { name: 'Org Admin', exact: true }).click()
      await page.getByRole('button', { name: 'Save', exact: true }).click()
      await expect(page.getByText('Org Admin request pending approval: register your own client sites here.')).toBeVisible()
      await page.getByRole('button', { name: 'Add Site', exact: true }).click()
      await page.getByLabel('Site Name', { exact: true }).fill(siteName)
      await page.getByRole('button', { name: 'Save', exact: true }).click()
      await expect(page.getByRole('dialog')).not.toBeVisible()
      await page.getByRole('button', { name: 'Next', exact: true }).click()
      await page.getByRole('button', { name: 'Go to my dashboard', exact: true }).click()
      await expect(page.getByText('Requested role: Org Admin')).toBeVisible()
      await expect(page.getByRole('button', { name: siteName, exact: true })).toBeVisible()
      await page.goto(NS + '/downloads')
      await expect(page.getByRole('button', { name: 'Download Startup Kit', exact: true })).toBeDisabled()
      await expect(page.getByRole('button', { name: 'Download FLARE Console', exact: true })).toBeDisabled()
      await login(admin)
      const userRow = admin.getByRole('row').filter({ hasText: email })
      await expect(userRow).toContainText('Org Admin (requested)')
      await userRow.getByRole('checkbox').click()
      await expect(userRow.getByRole('checkbox')).toBeChecked()
      await admin.getByRole('button', { name: 'Approve Selected', exact: true }).click()
      await expect(userRow).toContainText('Approved')
      await admin.goto(NS + '/site-dashboard')
      const siteRow = admin.getByRole('row').filter({ hasText: siteName })
      await siteRow.getByRole('checkbox').click()
      await expect(siteRow.getByRole('checkbox')).toBeChecked()
      await admin.getByRole('button', { name: 'Approve Selected', exact: true }).click()
      await expect(siteRow).toContainText('Approved')
      if (!frozenAtRegistration) {
        await admin.goto(NS + '/')
        await expect(admin.getByRole('button', { name: 'Freeze Project', exact: true })).toBeEnabled()
        await admin.getByRole('button', { name: 'Freeze Project', exact: true }).click()
        await admin.getByRole('dialog').getByRole('button', { name: 'Freeze', exact: true }).click()
        await expect(admin.getByText('Project status: Frozen')).toBeVisible()
      }
      await page.reload()
      const clientDownload = page.getByRole('button', { name: 'Download Startup Kit', exact: true })
      await expect(clientDownload).toBeEnabled()
      const session = await page.evaluate(() => JSON.parse(localStorage.getItem('session') || '{}'))
      const escalation = await page.request.patch(`${API}/users/${session.user.id}`, { headers: { Authorization: `Bearer ${session.user.token}` }, data: { role: 'project_admin' } })
      expect(escalation.status()).toBe(403)
      for (const kind of ['client', 'console']) {
        const button = kind === 'client' ? clientDownload : page.getByRole('button', { name: 'Download FLARE Console', exact: true })
        const pending = page.waitForEvent('download')
        await button.click()
        const downloaded = await pending
        const filename = testInfo.outputPath(`${kind}.zip`)
        await downloaded.saveAs(filename)
        const dialog = page.getByRole('dialog')
        await expect(dialog).toBeVisible()
        const pin = (await dialog.innerText()).match(/The secure PIN is:\s*([A-F0-9]+)/)?.[1]
        expect(pin).toBeTruthy()
        const metadata = JSON.parse(execFileSync('python3', [path.join(process.cwd(), 'tests/inspect_kit.py'), filename, pin!, kind, kind === 'client' ? siteName : email, 'Workshop_org'], { encoding: 'utf8' }))
        expect(metadata.encrypted).toBe(true)
        await dialog.getByRole('button', { name: 'Okay', exact: true }).click()
        await expect(dialog).toHaveCount(0)
      }
      await page.screenshot({ path: testInfo.outputPath('downloads.png'), fullPage: true })
      expect(failures).toEqual([])
    } finally {
      await adminContext.close()
    }
  })
}
