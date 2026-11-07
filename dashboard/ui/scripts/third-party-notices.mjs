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

import { existsSync, mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { formatNotices } from './compact-notices.mjs'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const defaultRoot = resolve(scriptDirectory, '..')
const licenseName = /^(?:licen[cs]e|copying|notice)(?:[.-].*)?$/i
const reviewedLicenses = new Set(['MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', '0BSD', 'MIT-0', '(MIT OR CC0-1.0)'])
const exactFallbacks = {
  'formik@2.4.9': 'formik-2.4.9-LICENSE.txt',
  'tiny-case@1.0.3': 'tiny-case-1.0.3-LICENSE.txt',
  'type-fest@2.19.0': 'type-fest-2.19.0-LICENSE.txt',
}

function readJson(filename) { return JSON.parse(readFileSync(filename, 'utf8')) }

function files(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    if (entry.isSymbolicLink()) throw new Error(`Unexpected symbolic link: ${join(directory, entry.name)}`)
    const filename = join(directory, entry.name)
    return entry.isDirectory() ? files(filename) : [filename]
  }).sort()
}

function within(parent, name) {
  const filename = resolve(parent, name)
  if (filename !== resolve(parent) && !filename.startsWith(resolve(parent) + sep)) {
    throw new Error(`Path escapes its directory: ${name}`)
  }
  return filename
}

function mapSources(map) {
  if (map.version !== 3 || map.sourceRoot) throw new Error('Unsupported browser source-map format.')
  if (map.sections) {
    const sources = map.sections.flatMap(section => mapSources(section.map))
    if (!sources.length) throw new Error('Browser source map has no attributable sources.')
    return sources
  }
  if (!Array.isArray(map.sources) || !map.sources.length || map.sources.some(source => typeof source !== 'string')) {
    throw new Error('Browser source map has no attributable sources.')
  }
  return map.sources
}

function mapReference(content) {
  return /(?:\r?\n)?\/\/# sourceMappingURL=(\S+)\s*$/.exec(content) ||
    /(?:\r?\n)?\/\*# sourceMappingURL=(\S+)\s*\*\/\s*$/.exec(content)
}

function generatedLoader(content, exportDirectory) {
  const match = /^__turbopack_load_page_chunks__\(("(?:[^"\\]|\\.)*"),\s*(\[[\s\S]*\])\)\s*;?\s*$/.exec(content)
  if (!match) return false
  const route = JSON.parse(match[1])
  const chunks = JSON.parse(match[2])
  return route.startsWith('/') && Array.isArray(chunks) && chunks.length > 0 && chunks.every(chunk =>
    typeof chunk === 'string' && chunk.startsWith('static/chunks/') && existsSync(within(join(exportDirectory, '_next'), chunk)))
}

/** Attribute the actual exported code; generated maps remain private build inputs. */
export function browserSources(root) {
  const exportDirectory = join(root, 'out')
  const buildId = readFileSync(join(root, '.next', 'BUILD_ID'), 'utf8').trim()
  const sources = new Set()
  const cleanups = []
  let mapped = 0
  for (const filename of files(exportDirectory)) {
    if (!/\.(?:js|css)$/.test(filename)) continue
    const name = relative(exportDirectory, filename).split(sep).join('/')
    if (!name.startsWith('_next/')) throw new Error(`Unmapped public code needs license review: ${name}`)
    const compiled = within(join(root, '.next'), name.slice('_next/'.length))
    const content = readFileSync(compiled, 'utf8')
    const reference = mapReference(content)
    const stripped = reference ? content.slice(0, reference.index) + '\n' : content
    const exported = readFileSync(filename, 'utf8')
    if (exported !== content && exported !== stripped) throw new Error(`Export differs from the mapped build: ${name}`)
    if (!reference) {
      const manifest = name.startsWith(`_next/static/${buildId}/`) &&
        ['_buildManifest.js', '_ssgManifest.js', '_clientMiddlewareManifest.js'].includes(name.split('/').at(-1))
      if (!manifest && !generatedLoader(content, exportDirectory)) {
        throw new Error(`Exported code has no source map: ${name}. Build with productionBrowserSourceMaps.`)
      }
      continue // These contain Next-generated route/asset data, covered by Next's license.
    }
    if (!/^[\w.-]+\.map$/.test(reference[1])) throw new Error(`Unsupported source-map URL: ${reference[1]}`)
    for (const source of mapSources(readJson(within(dirname(compiled), reference[1])))) sources.add(source)
    cleanups.push({ filename, content: stripped })
    mapped++
  }
  if (!mapped) throw new Error('No mapped browser assets found. Build the Dashboard first.')
  return { sources: [...sources].sort(), cleanups }
}

function licenseDocuments(root, sources) {
  const lock = readJson(join(root, 'package-lock.json'))
  if (lock.lockfileVersion !== 3 || !lock.packages) throw new Error('Expected an npm v3 package-lock.json.')
  const directories = new Map()
  const documents = []

  function addPackage(location) {
    if (directories.has(location)) return directories.get(location)
    const folder = within(root, location)
    const metadata = readJson(join(folder, 'package.json'))
    if (lock.packages[location]?.version !== metadata.version) throw new Error(`Dependency version differs from lockfile: ${location}`)
    const identity = `${metadata.name}@${metadata.version}`
    directories.set(location, { folder, identity })
    addDirectory(folder, identity, metadata.license, metadata)
    return { folder, identity }
  }

  function addDirectory(folder, identity, license, metadata) {
    if (!reviewedLicenses.has(license)) throw new Error(`Unreviewed distributed license ${String(license)}: ${identity}`)
    const names = readdirSync(folder, { withFileTypes: true }).filter(entry => entry.isFile() && licenseName.test(entry.name)).map(entry => entry.name).sort()
    let notices = names.map(filename => ({ filename, text: readFileSync(join(folder, filename), 'utf8').trim() }))
    if (!notices.length) {
      for (const filename of ['README.md', 'readme.md', 'README']) {
        if (!existsSync(join(folder, filename))) continue
        const text = readFileSync(join(folder, filename), 'utf8')
        const section = [...text.matchAll(/^\s*(?:#{1,6}\s*)?Licen[cs]e\s*\n(?:[-=]+\s*\n)?/gim)].at(-1)
        const body = section ? text.slice(section.index + section[0].length).split(/^\[[^\]]+\]:/m)[0].trim() : ''
        if (/Permission is hereby granted|Apache License/i.test(body)) { notices = [{ filename: `${filename} (license section)`, text: body }]; break }
      }
    }
    const fallback = exactFallbacks[identity]
    if (!notices.length && fallback) notices = [{ filename: fallback, text: readFileSync(join(scriptDirectory, 'licenses', fallback), 'utf8').trim() }]
    if (!notices.length && metadata.name === 'client-only' && metadata.version === '0.0.1') {
      notices = [{ filename: 'react/LICENSE', text: readFileSync(join(root, 'node_modules/react/LICENSE'), 'utf8').trim() }]
    }
    if (!notices.length) throw new Error(`No license text found for ${identity}. Add a reviewed, version-specific fallback.`)
    for (const notice of notices) documents.push({ identity, license, ...notice })
  }

  addPackage('node_modules/next') // Covers the generated loader/runtime and route manifests too.
  const seenDirectories = new Set()
  for (const source of sources) {
    if (/^turbopack:\/\/\/\[(?:next|turbopack)\]\//.test(source)) continue
    const prefix = 'turbopack:///[project]/'
    if (!source.startsWith(prefix)) throw new Error(`Unrecognized browser source needs license review: ${source}`)
    const path = source.slice(prefix.length)
    if (!path.startsWith('node_modules/')) {
      if (!/^(?:src\/|styles\/|siteparams\.json(?:$|\.))/.test(path)) throw new Error(`Unrecognized project source: ${source}`)
      continue
    }
    const segments = path.split('/')
    const offset = segments.lastIndexOf('node_modules')
    const end = offset + (segments[offset + 1].startsWith('@') ? 3 : 2)
    const location = segments.slice(0, end).join('/')
    const { folder, identity } = addPackage(location)
    const modulePath = within(root, path)
    let parent = dirname(modulePath)
    while (parent !== folder) {
      if (seenDirectories.has(parent)) break
      seenDirectories.add(parent)
      if (existsSync(parent)) {
        const entries = readdirSync(parent)
        const packageFile = join(parent, 'package.json')
        const metadata = existsSync(packageFile) ? readJson(packageFile) : undefined
        if (entries.some(name => licenseName.test(name)) || metadata?.name || metadata?.license) {
          if (!metadata) throw new Error(`Vendored license needs metadata review: ${parent}`)
          addDirectory(parent, `${identity}/${relative(folder, parent).split(sep).join('/')}`, metadata.license, metadata)
        }
      }
      parent = dirname(parent)
      if (!parent.startsWith(folder + sep) && parent !== folder) throw new Error(`Source escapes package: ${source}`)
    }
  }
  return documents
}

export function collectNotices(root) {
  return formatNotices(licenseDocuments(root, browserSources(root).sources))
}

export function generateNotices(root, output = join(root, 'out', 'THIRD_PARTY_NOTICES.txt')) {
  const { sources, cleanups } = browserSources(root)
  const text = formatNotices(licenseDocuments(root, sources))
  mkdirSync(dirname(output), { recursive: true })
  writeFileSync(output, text)
  for (const cleanup of cleanups) writeFileSync(cleanup.filename, cleanup.content)
  for (const filename of files(join(root, 'out'))) if (filename.endsWith('.map')) unlinkSync(filename)
  return text
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2)
  if (args.length && (args.length !== 2 || args[0] !== '--output')) throw new Error('Usage: node scripts/third-party-notices.mjs [--output FILE]')
  const output = args.length ? resolve(args[1]) : join(defaultRoot, 'out', 'THIRD_PARTY_NOTICES.txt')
  const text = generateNotices(defaultRoot, output)
  console.log(`Wrote ${relative(defaultRoot, output)} (${Buffer.byteLength(text)} bytes); removed private browser source maps.`)
}
