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

import nextConfig from '../../next.config'
import { PHASE_DEVELOPMENT_SERVER, PHASE_PRODUCTION_BUILD } from 'next/constants'

const original = process.env.NVFLARE_DASHBOARD_BACKEND_URL

afterEach(() => {
  if (original === undefined) delete process.env.NVFLARE_DASHBOARD_BACKEND_URL
  else process.env.NVFLARE_DASHBOARD_BACKEND_URL = original
})

test('production export keeps the Flask prefix and has no runtime proxy', () => {
  process.env.NVFLARE_DASHBOARD_BACKEND_URL = 'http://127.0.0.1:8443'
  const config = nextConfig(PHASE_PRODUCTION_BUILD)
  expect(config.output).toBe('export')
  expect(config.basePath).toBe('/nvflare-dashboard')
  expect(config.rewrites).toBeUndefined()
})

test('development proxies the API without changing the production API prefix', async () => {
  process.env.NVFLARE_DASHBOARD_BACKEND_URL = 'http://127.0.0.1:8443/'
  const config = nextConfig(PHASE_DEVELOPMENT_SERVER)
  expect(config.output).toBeUndefined()
  expect(await config.rewrites?.()).toEqual([{ source: '/api/v1/:path*', destination: 'http://127.0.0.1:8443/nvflare-dashboard/api/v1/:path*' }])
})
