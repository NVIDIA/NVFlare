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

import { expect, test } from '@playwright/test'
import { Project } from '../../src/types'

const NS = '/nvflare-dashboard'

test('administrators must publish a project before freezing it', async ({ page }) => {
  let project: Project = {
    short_name: 'Example_Project', title: 'Example Project', description: '', app_location: '',
    public: false, frozen: false, server1: 'server.example.test',
  }
  const projectPatches: Partial<Project>[] = []
  await page.addInitScript(() => localStorage.setItem('session', JSON.stringify({
    user: { email: 'admin@example.test', id: 1, role: 4, token: 'token' }, status: 'authenticated',
  })))
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const resource = new URL(request.url()).pathname.split('/api/v1/')[1]
    if (resource === 'project') {
      if (request.method() === 'PATCH') {
        const patch = request.postDataJSON() as Partial<Project>
        projectPatches.push(patch)
        project = { ...project, ...patch }
      }
      return route.fulfill({ json: { status: 'ok', project } })
    }
    if (resource === 'clients') return route.fulfill({ json: { status: 'ok', client_list: [] } })
    throw new Error(`Unexpected request: ${request.method()} ${resource}`)
  })

  await page.goto(NS + '/')
  await expect(page.getByRole('alert').filter({ hasText: 'Make the project public in Project Configuration before freezing it.' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Freeze Project', exact: true })).toBeDisabled()
  expect(projectPatches).toEqual([])

  await page.getByRole('main').getByRole('link', { name: 'Project Configuration', exact: true }).click()
  await page.getByRole('button', { name: 'Make Project Public', exact: true }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'Project settings saved.' })).toBeVisible()
  expect(projectPatches).toEqual([{ public: true }])
  await page.getByRole('link', { name: 'Project Home', exact: true }).click()

  const freezeButton = page.getByRole('button', { name: 'Freeze Project', exact: true })
  await expect(freezeButton).toBeEnabled()
  await freezeButton.click()
  await page.getByRole('dialog').getByRole('button', { name: 'Freeze', exact: true }).click()
  await expect(page.getByText('Project status: Frozen', { exact: true })).toBeVisible()
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await expect(freezeButton).toBeDisabled()
  expect(projectPatches).toEqual([{ public: true }, { frozen: true }])
})
