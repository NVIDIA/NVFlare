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

const NS = '/nvflare-dashboard'
async function mockProject(page: Page) {
  await page.route('**/api/v1/**', async route => {
    const resource = new URL(route.request().url()).pathname.split('/api/v1/')[1]
    if (resource === 'project') return route.fulfill({ json: { status: 'ok', project: { short_name: 'test', title: 'Workshop', description: '', app_location: '', public: true, frozen: false } } })
    if (resource === 'login') return route.fulfill({ status: 401, json: { status: 'unauthenticated' } })
    if (resource === 'users') return route.fulfill({ json: { status: 'ok', user_list: [] } })
    if (resource === 'clients') return route.fulfill({ json: { status: 'ok', client_list: [] } })
    throw new Error(`Unexpected API resource: ${resource}`)
  })
}

test('invalid credentials show login guidance without claiming session expiry', async ({ page }) => {
  await mockProject(page)
  await page.goto(NS + '/')
  await page.getByRole('textbox', { name: /^Email/ }).fill('wrong@example.test')
  await page.getByLabel(/^Password/).fill('incorrect')
  await page.getByRole('button', { name: 'Log in', exact: true }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'Unable to log in. Check your email and password.' })).toBeVisible()
  await expect(page.getByText('Your session has expired. Please log in again.', { exact: true })).not.toBeVisible()
  await expect(page.getByRole('link', { name: 'Sign Up', exact: true })).toBeVisible()
})

test('mobile navigation opens and routes through the dashboard prefix', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.addInitScript(() => localStorage.setItem('session', JSON.stringify({ user: { email: 'admin@example.test', id: 1, role: 4, token: 'token' }, status: 'authenticated' })))
  await mockProject(page)
  await page.goto(NS + '/')
  await page.getByRole('button', { name: 'Open navigation' }).click()
  await page.getByRole('link', { name: 'Users Dashboard', exact: true }).click()
  await expect(page).toHaveURL(/\/nvflare-dashboard\/project-admin-dashboard$/)
  await expect(page.getByRole('heading', { name: 'Users Dashboard' })).toBeVisible()
})
