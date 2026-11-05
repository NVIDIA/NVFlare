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

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const defaultRoot = resolve(scriptDirectory, '..')
const licenseName = /^(?:licen[cs]e|copying|notice)(?:[.-].*)?$/i
const exactFallbacks = {
  'formik@2.4.9': {
    file: 'formik-2.4.9-LICENSE.txt',
    source: 'https://github.com/jaredpalmer/formik/blob/main/LICENSE',
  },
  'tiny-case@1.0.3': {
    file: 'tiny-case-1.0.3-LICENSE.txt',
    source: 'Published package.json license and author metadata; standard MIT text.',
  },
  'type-fest@2.19.0': {
    file: 'type-fest-2.19.0-LICENSE.txt',
    source: 'https://github.com/sindresorhus/type-fest/blob/main/license-mit',
  },
}

function readJson(filename) {
  return JSON.parse(readFileSync(filename, 'utf8'))
}

function licenseFiles(folder, subdirectory = '') {
  const files = []
  const entries = readdirSync(join(folder, subdirectory), { withFileTypes: true })
  for (const entry of entries) {
    if (entry.name === 'node_modules' || entry.name === '.git') continue
    const filename = join(subdirectory, entry.name)
    if (entry.isDirectory()) files.push(...licenseFiles(folder, filename))
    else if (entry.isFile() && licenseName.test(entry.name)) files.push(filename)
  }
  return files.sort()
}

function readmeLicense(folder) {
  for (const filename of ['README.md', 'readme.md', 'README']) {
    if (!existsSync(join(folder, filename))) continue
    const text = readFileSync(join(folder, filename), 'utf8')
    const sections = [...text.matchAll(/^\s*(?:#{1,6}\s*)?Licen[cs]e\s*\n(?:[-=]+\s*\n)?/gim)]
    const last = sections.at(-1)
    if (!last) continue
    const body = text.slice(last.index + last[0].length).split(/^\[[^\]]+\]:/m)[0].trim()
    if (/Permission is hereby granted|Apache License|GNU (?:LESSER |GENERAL )?PUBLIC LICENSE/i.test(body)) {
      return [{ filename: `${filename} (license section)`, text: body }]
    }
  }
  return []
}

function fallbackNotices(root, metadata) {
  const key = `${metadata.name}@${metadata.version}`
  const exact = exactFallbacks[key]
  if (exact) return [{ filename: `${exact.file}; source: ${exact.source}`, text: readFileSync(join(scriptDirectory, 'licenses', exact.file), 'utf8').trim() }]
  if (metadata.name === '@next/env' || metadata.name.startsWith('@next/swc-')) {
    const next = readJson(join(root, 'node_modules', 'next', 'package.json'))
    if (next.version === metadata.version) return [{ filename: 'next/license.md (same-version Next.js package)', text: readFileSync(join(root, 'node_modules', 'next', 'license.md'), 'utf8').trim() }]
  }
  if (key === 'client-only@0.0.1') {
    return [{ filename: 'react/LICENSE (React marker package)', text: readFileSync(join(root, 'node_modules', 'react', 'LICENSE'), 'utf8').trim() }]
  }
  return []
}

/** Read dependency metadata and notice text only; never import dependency code. */
export function collectNotices(root) {
  const lock = readJson(join(root, 'package-lock.json'))
  if (lock.lockfileVersion !== 3 || !lock.packages) throw new Error('Expected an npm v3 package-lock.json.')
  const sections = []
  const seen = new Set()
  for (const location of Object.keys(lock.packages).sort()) {
    const locked = lock.packages[location]
    if (!location || locked.dev) continue
    if (!location.startsWith('node_modules/') || location.split('/').includes('..')) {
      throw new Error(`Unsupported package location: ${location}`)
    }
    const folder = join(root, location)
    if (!existsSync(folder)) {
      if (locked.optional) continue // Platform-specific build dependencies may be absent.
      throw new Error(`Required dependency is not installed: ${location}. Run npm ci.`)
    }
    const metadata = readJson(join(folder, 'package.json'))
    if (metadata.version !== locked.version) throw new Error(`Dependency version differs from lockfile: ${location}`)
    // Native compiler/image binaries are build tools, never exported browser code.
    if (metadata.name.startsWith('@img/') || metadata.name.startsWith('@next/swc-')) continue
    const identity = `${metadata.name}@${metadata.version}`
    if (seen.has(identity)) continue
    seen.add(identity)
    const files = licenseFiles(folder)
    let notices = files.map(filename => ({ filename, text: readFileSync(join(folder, filename), 'utf8').trim() }))
    if (!notices.length) notices = readmeLicense(folder)
    if (!notices.length) notices = fallbackNotices(root, metadata)
    if (!notices.length) throw new Error(`No license text found for ${identity}. Add a reviewed, version-specific fallback.`)
    const repository = typeof metadata.repository === 'string' ? metadata.repository : metadata.repository?.url
    const text = [
      '='.repeat(78), identity,
      `Declared license: ${metadata.license || locked.license || 'See notice text below'}`,
      ...(repository ? [`Package source: ${repository}`] : []),
      ...notices.flatMap(notice => [`\n--- ${notice.filename.replaceAll('\\', '/')} ---\n`, notice.text]),
    ].join('\n')
    sections.push({ identity, text })
  }
  sections.sort((left, right) => left.identity < right.identity ? -1 : left.identity > right.identity ? 1 : 0)
  return [
    'NVFlare Dashboard third-party notices',
    '',
    'This inventory retains notices from installed production dependencies, including',
    'build-time members of that dependency tree. Some listed code is not included in',
    'the static browser assets. Native build tools and node_modules are not shipped',
    'with the exported Dashboard. Native @img and @next/swc packages, and all',
    'development-only dependencies, are therefore excluded from this inventory.',
    '',
    ...sections.map(section => section.text),
    '',
  ].join('\n')
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const arguments_ = process.argv.slice(2)
  if (arguments_.length && (arguments_.length !== 2 || arguments_[0] !== '--output')) {
    throw new Error('Usage: node scripts/third-party-notices.mjs [--output FILE]')
  }
  const output = arguments_.length ? resolve(arguments_[1]) : join(defaultRoot, 'out', 'THIRD_PARTY_NOTICES.txt')
  const text = collectNotices(defaultRoot)
  mkdirSync(dirname(output), { recursive: true })
  writeFileSync(output, text)
  console.log(`Wrote ${relative(defaultRoot, output)} (${text.length} characters).`)
}
