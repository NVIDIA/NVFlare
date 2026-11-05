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

import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import assert from 'node:assert/strict'
import test from 'node:test'
import { collectNotices } from './third-party-notices.mjs'

function fixture(t, packages) {
  const root = mkdtempSync(join(tmpdir(), 'nvflare-notices-'))
  t.after(() => rmSync(root, { recursive: true, force: true }))
  writeFileSync(join(root, 'package-lock.json'), JSON.stringify({ lockfileVersion: 3, packages: { '': {}, ...packages } }))
  return root
}

function install(root, name, version, files = {}) {
  const folder = join(root, 'node_modules', name)
  mkdirSync(folder, { recursive: true })
  writeFileSync(join(folder, 'package.json'), JSON.stringify({ name, version, license: 'MIT' }))
  writeFileSync(join(folder, 'index.js'), 'throw new Error("Dependency code must never execute")')
  for (const [filename, text] of Object.entries(files)) {
    mkdirSync(join(folder, filename, '..'), { recursive: true })
    writeFileSync(join(folder, filename), text)
  }
}

test('retains runtime and vendored notices without executing packages', t => {
  const root = fixture(t, {
    'node_modules/z-runtime': { version: '1.0.0' },
    'node_modules/a-runtime': { version: '2.0.0' },
    'node_modules/dev-only': { version: '1.0.0', dev: true },
    'node_modules/platform-only': { version: '1.0.0', optional: true },
  })
  install(root, 'z-runtime', '1.0.0', { LICENSE: 'Copyright Runtime Z\nMIT license text', 'vendor/NOTICE': 'Vendor attribution' })
  install(root, 'a-runtime', '2.0.0', { 'LICENSE.md': 'Copyright Runtime A\nMIT license text' })
  const output = collectNotices(root)
  assert.ok(output.includes('Copyright Runtime Z'))
  assert.ok(output.includes('Vendor attribution'))
  assert.ok(output.indexOf('a-runtime@2.0.0') < output.indexOf('z-runtime@1.0.0'))
  assert.equal(output.includes('dev-only@'), false)
  assert.equal(output.includes('platform-only@'), false)
  assert.equal(collectNotices(root), output)
})

test('fails when a required installed package has no reviewed notice', t => {
  const root = fixture(t, { 'node_modules/unlicensed-fixture': { version: '1.0.0' } })
  install(root, 'unlicensed-fixture', '1.0.0')
  assert.throws(() => collectNotices(root), /No license text found for unlicensed-fixture@1.0.0/)
})

test('rejects an installed version that differs from the lockfile', t => {
  const root = fixture(t, { 'node_modules/runtime': { version: '1.0.0' } })
  install(root, 'runtime', '2.0.0', { LICENSE: 'MIT' })
  assert.throws(() => collectNotices(root), /Dependency version differs from lockfile/)
})
