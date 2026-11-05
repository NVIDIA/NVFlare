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

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { formatNotices } from './compact-notices.mjs'

const mit = readFileSync(new URL('./licenses/shared-MIT.txt', import.meta.url), 'utf8').trim()
const apache = readFileSync(new URL('./licenses/shared-Apache-2.0.txt', import.meta.url), 'utf8').trim()
const notice = (identity, text, filename = 'LICENSE', license = 'MIT') => ({ identity, filename, text, license })
const count = (text, fragment) => text.split(fragment).length - 1

test('shares wrapped MIT text and attribution while retaining every package/file mapping', () => {
  const copyright = 'MIT License\n\nCopyright (c) 2026 Shared Fixture'
  const wrapped = mit.split(/\s+/u).join('\n')
  const output = formatNotices([
    notice('b-package@2.0.0', `${copyright}\n\n${wrapped}`, 'vendor/LICENSE'),
    notice('a-package@1.0.0', `${copyright}\n\n${mit}`),
  ])
  assert.equal(count(output, 'Permission is hereby granted'), 1)
  assert.equal(count(output, 'Copyright (c) 2026 Shared Fixture'), 1)
  assert.ok(output.includes('a-package@1.0.0 / LICENSE'))
  assert.ok(output.includes('b-package@2.0.0 / vendor/LICENSE'))
  assert.ok(output.indexOf('a-package@') < output.indexOf('b-package@'))
  assert.ok(output.includes('[MIT]\n' + mit))
})

test('keeps different copyrights while sharing their complete MIT body', () => {
  const output = formatNotices([
    notice('first@1', `Copyright First\n\n${mit}`),
    notice('second@1', `Copyright Second\n\n${mit}`),
  ])
  assert.equal(count(output, 'Permission is hereby granted'), 1)
  assert.ok(output.includes('Copyright First'))
  assert.ok(output.includes('Copyright Second'))
})

test('modified MIT clauses remain complete and unshared', () => {
  const changed = mit.replace('THE SOFTWARE IS PROVIDED', 'An additional distribution condition applies.\n\nTHE SOFTWARE IS PROVIDED')
  const original = `Copyright Special Fixture\n\n${changed}`
  const output = formatNotices([notice('changed@1', original)])
  assert.ok(output.includes(original))
  assert.equal(output.includes('[MIT]'), false)
})

test('retains multiple grants, their order, and appended NOTICE conditions', () => {
  const output = formatNotices([notice('compound@1', [
    'Copyright First Author', mit,
    'Second included work: Copyright Second Author', mit,
    'NOTICE: An extra condition follows both grants.\nKeep this attribution unchanged.',
  ].join('\n\n'))])
  assert.equal(count(output, 'Permission is hereby granted'), 1)
  assert.ok(output.includes('Text sections in order: N1 + MIT + N2 + MIT + N3'))
  assert.ok(output.includes('Copyright First Author'))
  assert.ok(output.includes('Second included work: Copyright Second Author'))
  assert.ok(output.includes('NOTICE: An extra condition follows both grants.\nKeep this attribution unchanged.'))
})

test('retains Apache appendices, actual copyrights, and NOTICE files', () => {
  const appendix = 'APPENDIX: Application to this work.\nCopyright (c) 2026 Apache Fixture\nAll additional notices apply.'
  const output = formatNotices([
    notice('apache@1', `${apache}\n\n${appendix}`, 'LICENSE', 'Apache-2.0'),
    notice('apache@1', 'NOTICE\nOriginal work by Fixture Contributors.', 'NOTICE', 'Apache-2.0'),
    notice('apache-copy@2', apache.split(/\s+/u).join('  '), 'LICENSE', 'Apache-2.0'),
  ])
  assert.equal(count(output, 'END OF TERMS AND CONDITIONS'), 1)
  assert.ok(output.includes('[Apache-2.0]\n' + apache))
  assert.ok(output.includes(appendix))
  assert.ok(output.includes('NOTICE\nOriginal work by Fixture Contributors.'))
  assert.ok(output.includes('apache@1 / NOTICE'))
})

test('unknown licenses stay intact and only identical text is pooled', () => {
  const custom = 'Custom License\nCopyright Unknown Fixture\nSpecial words, punctuation; and conditions must remain.'
  const output = formatNotices([
    notice('custom-a@1', custom, 'COPYING', 'Custom'),
    notice('custom-b@2', custom.replaceAll('\n', '\r\n'), 'LICENSE', 'Custom'),
  ])
  assert.ok(output.includes(custom))
  assert.equal(count(output, 'Copyright Unknown Fixture'), 1)
  assert.ok(output.includes('custom-a@1 / COPYING'))
  assert.ok(output.includes('custom-b@2 / LICENSE'))
  assert.equal(output.includes('[MIT]'), false)
  assert.equal(output.includes('[Apache-2.0]'), false)
})

test('appended conditions remain after a recognized shared body', () => {
  const condition = 'ADDITIONAL CONDITION: Do not remove this condition or its attribution.'
  const output = formatNotices([notice('extra@1', `Copyright Extra\n\n${mit}\n\n${condition}`)])
  assert.ok(output.includes('Text sections in order: N1 + MIT + N2'))
  assert.ok(output.includes(condition))
})

test('format is deterministic and rejects conflicting source documents', () => {
  const inputs = [notice('z@1', 'Unknown terms Z'), notice('a@1', 'Unknown terms A')]
  assert.equal(formatNotices(inputs), formatNotices(inputs.toReversed()))
  assert.throws(() => formatNotices([
    notice('same@1', 'Terms One'), notice('same@1', 'Terms Two'),
  ]), /Conflicting notice documents/)
})
