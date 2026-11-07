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

import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'
import test from 'node:test'
import { browserSources, collectNotices, generateNotices } from './third-party-notices.mjs'

function put(filename, content) {
  mkdirSync(dirname(filename), { recursive: true })
  writeFileSync(filename, content)
}

function install(root, location, version, files = { LICENSE: 'Copyright Fixture\nMIT license text' }, license = 'MIT') {
  const name = location.split('node_modules/').at(-1)
  const folder = join(root, location)
  put(join(folder, 'package.json'), JSON.stringify({ name, version, license }))
  put(join(folder, 'index.js'), 'throw new Error("Dependency code must never execute")')
  for (const [filename, text] of Object.entries(files)) put(join(folder, filename), text)
}

function fixture(t, packages) {
  const root = mkdtempSync(join(tmpdir(), 'nvflare-notices-'))
  t.after(() => rmSync(root, { recursive: true, force: true }))
  put(join(root, 'package-lock.json'), JSON.stringify({ lockfileVersion: 3, packages: {
    '': {}, 'node_modules/next': { version: '16.4.5' }, ...packages,
  } }))
  put(join(root, '.next/BUILD_ID'), 'test-build')
  install(root, 'node_modules/next', '16.4.5', { LICENSE: 'Copyright Next Fixture\nMIT license text' })
  return root
}

function asset(root, sources, name = 'static/chunks/test.js', options = {}) {
  const body = options.body || 'globalThis.testFixture = true;'
  const url = options.url || 'private.map'
  const content = options.unmapped ? body : `${body}\n//# sourceMappingURL=${url}\n`
  put(join(root, '.next', name), content)
  put(join(root, 'out/_next', name), content)
  if (!options.unmapped) {
    const map = options.map || { version: 3, sources, mappings: '' }
    put(join(root, '.next', dirname(name), url), JSON.stringify(map))
    put(join(root, 'out/_next', dirname(name), url), JSON.stringify(map))
  }
  return join(root, 'out/_next', name)
}

const source = path => `turbopack:///[project]/${path}`

test('includes mapped packages and vendored notices, excluding installed build/server/dev code', t => {
  const root = fixture(t, {
    'node_modules/browser': { version: '1.0.0' },
    'node_modules/build-only': { version: '1.0.0' },
    'node_modules/dev-only': { version: '1.0.0', dev: true },
  })
  install(root, 'node_modules/browser', '1.0.0', { LICENSE: 'Copyright Browser', NOTICE: 'Keep Browser attribution' })
  install(root, 'node_modules/build-only', '1.0.0', { LICENSE: 'Build-only license' }, 'LGPL-3.0-or-later')
  install(root, 'node_modules/dev-only', '1.0.0')
  put(join(root, 'node_modules/next/dist/compiled/used/package.json'), JSON.stringify({ name: 'used', license: 'MIT' }))
  put(join(root, 'node_modules/next/dist/compiled/used/LICENSE'), 'Copyright Used Vendor')
  put(join(root, 'node_modules/next/dist/compiled/unused/LICENSE'), 'Unused MPL license')
  asset(root, [source('node_modules/browser/index.js'), source('node_modules/next/dist/compiled/used/index.js')])
  const output = collectNotices(root)
  for (const text of ['Copyright Browser', 'Keep Browser attribution', 'Copyright Used Vendor', 'Copyright Next Fixture']) assert.ok(output.includes(text))
  for (const text of ['build-only', 'dev-only', 'Unused MPL license']) assert.equal(output.includes(text), false)
})

test('retains nested versions of a package and sectioned source maps', t => {
  const root = fixture(t, {
    'node_modules/identity': { version: '2.0.0' },
    'node_modules/parent/node_modules/identity': { version: '1.0.0' },
  })
  install(root, 'node_modules/identity', '2.0.0')
  install(root, 'node_modules/parent/node_modules/identity', '1.0.0')
  asset(root, [], 'static/chunks/test.js', { map: { version: 3, sections: [
    { offset: { line: 0, column: 0 }, map: { version: 3, sources: [source('node_modules/identity/index.js')], mappings: '' } },
    { offset: { line: 1, column: 0 }, map: { version: 3, sources: [source('node_modules/parent/node_modules/identity/index.js')], mappings: '' } },
  ] } })
  const output = collectNotices(root)
  assert.ok(output.includes('identity@1.0.0'))
  assert.ok(output.includes('identity@2.0.0'))
})

test('rejects empty ordinary and indexed source maps', t => {
  for (const map of [{ version: 3, sources: [], mappings: '' }, { version: 3, sections: [] }]) {
    const root = fixture(t, {})
    asset(root, [], 'static/chunks/test.js', { map })
    assert.throws(() => collectNotices(root), /no attributable sources/)
  }
})

test('private maps and their URL comments are removed; generation remains repeatable', t => {
  const root = fixture(t, {})
  const filename = asset(root, [source('node_modules/next/dist/client/index.js')])
  const text = generateNotices(root)
  assert.equal(readFileSync(filename, 'utf8'), 'globalThis.testFixture = true;\n')
  assert.equal(existsSync(join(root, 'out/_next/static/chunks/private.map')), false)
  assert.equal(existsSync(join(root, '.next/static/chunks/private.map')), true)
  assert.equal(generateNotices(root), text)
})

test('unmapped generated page-loader data is accepted but arbitrary executable code is rejected', t => {
  const root = fixture(t, {})
  asset(root, ['turbopack:///[turbopack]/runtime.ts'])
  asset(root, [], 'static/chunks/loader.js', { unmapped: true, body: '__turbopack_load_page_chunks__("/", ["static/chunks/test.js"])' })
  assert.ok(collectNotices(root).includes('Copyright Next Fixture'))
  asset(root, [], 'static/chunks/unknown.js', { unmapped: true, body: 'arbitraryUnattributedCode()' })
  assert.throws(() => collectNotices(root), /no source map/)
})

test('fails closed for missing attribution, changed package versions, and unreviewed licenses', t => {
  for (const scenario of ['missing', 'version', 'license']) {
    const root = fixture(t, { 'node_modules/runtime': { version: '1.0.0' } })
    install(root, 'node_modules/runtime', scenario === 'version' ? '2.0.0' : '1.0.0', scenario === 'missing' ? {} : undefined, scenario === 'license' ? 'GPL-3.0-only' : 'MIT')
    asset(root, [source('node_modules/runtime/index.js')])
    assert.throws(() => collectNotices(root), scenario === 'missing' ? /No license text/ : scenario === 'version' ? /version differs/ : /Unreviewed distributed license/)
  }
})

test('vendored restrictive licenses cannot hide behind the parent package MIT license', t => {
  const root = fixture(t, {})
  put(join(root, 'node_modules/next/dist/compiled/copyleft/package.json'), JSON.stringify({ name: 'copyleft', license: 'MPL-2.0' }))
  put(join(root, 'node_modules/next/dist/compiled/copyleft/LICENSE'), 'Mozilla Public License Version 2.0')
  asset(root, [source('node_modules/next/dist/compiled/copyleft/index.js')])
  assert.throws(() => collectNotices(root), /Unreviewed distributed license MPL-2.0/)
})

test('vendored metadata requires review even when its license file is missing', t => {
  for (const license of ['MPL-2.0', 'MIT']) {
    const root = fixture(t, {})
    put(join(root, 'node_modules/next/dist/compiled/missing/package.json'), JSON.stringify({ name: 'missing', license }))
    asset(root, [source('node_modules/next/dist/compiled/missing/index.js')])
    assert.throws(() => collectNotices(root), license === 'MIT' ? /No license text/ : /Unreviewed distributed license/)
  }
})

test('rejects tampered exports, unknown source namespaces, and escaping map URLs', t => {
  const changed = fixture(t, {})
  const filename = asset(changed, [source('node_modules/next/index.js')])
  writeFileSync(filename, 'different export')
  assert.throws(() => collectNotices(changed), /Export differs/)
  const unknown = fixture(t, {})
  asset(unknown, ['https://unknown.example/library.js'])
  assert.throws(() => collectNotices(unknown), /Unrecognized browser source/)
  const escaping = fixture(t, {})
  asset(escaping, [source('node_modules/next/index.js')], 'static/chunks/escape.js', { url: '../outside.map' })
  assert.throws(() => browserSources(escaping), /Unsupported source-map URL/)
})
