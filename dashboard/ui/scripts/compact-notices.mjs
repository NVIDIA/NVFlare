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

import { readFileSync } from 'node:fs'

const normalize = text => text.replace(/\s+/gu, ' ').trim()
const escape = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
const compare = (left, right) => left < right ? -1 : left > right ? 1 : 0

const shared = [
  ['MIT', 'shared-MIT.txt'],
  ['Apache-2.0', 'shared-Apache-2.0.txt'],
].map(([id, filename]) => {
  const text = readFileSync(new URL(`./licenses/${filename}`, import.meta.url), 'utf8').trim()
  if (!text) throw new Error(`Shared license text is empty: ${filename}`)
  // Only whitespace may vary. Changes to words, case, or punctuation stay intact.
  const expression = `(?<!\\S)${text.split(/\s+/u).map(escape).join('\\s+')}(?!\\S)`
  return { id, text, expression }
})

function splitNotice(text) {
  const matches = shared.flatMap(license =>
    [...text.matchAll(new RegExp(license.expression, 'gu'))].map(match => ({
      start: match.index, end: match.index + match[0].length, license,
    })))
  matches.sort((left, right) => left.start - right.start || left.end - right.end)
  const pieces = []
  let cursor = 0
  for (const match of matches) {
    if (match.start < cursor) throw new Error('Overlapping shared license texts.')
    if (normalize(text.slice(cursor, match.start))) pieces.push({ text: text.slice(cursor, match.start).trim() })
    pieces.push({ id: match.license.id, text: match.license.text })
    cursor = match.end
  }
  if (normalize(text.slice(cursor))) pieces.push({ text: text.slice(cursor).trim() })
  if (normalize(pieces.map(piece => piece.text).join('\n\n')) !== normalize(text)) {
    throw new Error('Compact notice reconstruction differs from its source.')
  }
  return pieces
}

/** Share complete approved license bodies without interpreting attribution text. */
export function formatNotices(documents) {
  const texts = new Map(shared.map(license => [license.id, license.text]))
  const identities = new Map(shared.map(license => [normalize(license.text), license.id]))
  const used = new Set()
  const groups = new Map()
  const sources = new Map()
  let counter = 0
  const ordered = [...documents].sort((left, right) =>
    compare(left.identity, right.identity) || compare(left.filename, right.filename))
  for (const document of ordered) {
    if (![document.identity, document.filename, document.text].every(value => typeof value === 'string' && value.trim())) {
      throw new Error('Each notice requires a nonempty identity, filename, and text.')
    }
    const source = JSON.stringify([document.identity, document.filename])
    const original = normalize(document.text)
    if (sources.has(source) && sources.get(source) !== original) {
      throw new Error(`Conflicting notice documents: ${document.identity} / ${document.filename}`)
    }
    sources.set(source, original)
    const ids = splitNotice(document.text).map(piece => {
      let id = piece.id || identities.get(normalize(piece.text))
      if (!id) {
        id = `N${++counter}`
        identities.set(normalize(piece.text), id)
        texts.set(id, piece.text)
      }
      used.add(id)
      return id
    })
    // Recheck after pooling: deduplication may change whitespace, never content.
    if (normalize(ids.map(id => texts.get(id)).join('\n\n')) !== original) {
      throw new Error(`Pooled notice reconstruction differs: ${document.identity} / ${document.filename}`)
    }
    const references = ids.join(' + ')
    const mapping = `${document.identity} / ${document.filename}${document.license ? ` (declared ${document.license})` : ''}`
    if (!groups.has(references)) groups.set(references, new Set())
    groups.get(references).add(mapping)
  }
  const index = [...groups].flatMap(([references, mappings]) => [
    `Text sections in order: ${references}`,
    ...[...mappings].map(mapping => `  ${mapping}`),
    '',
  ])
  const definitions = [...texts].filter(([id]) => used.has(id))
  // Put attributions and exceptions before the common license-text appendix.
  const bodies = [
    ...definitions.filter(([id]) => id.startsWith('N')),
    ...definitions.filter(([id]) => !id.startsWith('N')),
  ].flatMap(([id, text]) => [
    `[${id}]`, text, '',
  ])
  return [
    'NVFlare Dashboard third-party notices',
    '',
    'Package/file mappings refer to the text sections below, in the stated order.',
    'Identical text is included once. Shared license bodies preserve all original',
    'words and punctuation; surrounding attributions and conditions remain intact.',
    'Declared package licenses do not replace the supplied license/NOTICE text.',
    '',
    ...index,
    ...bodies,
  ].join('\n')
}
