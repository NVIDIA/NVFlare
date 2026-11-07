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

const { PHASE_DEVELOPMENT_SERVER } = require('next/constants')

/** @param {string} phase @returns {import('next').NextConfig} */
module.exports = phase => {
  const development = phase === PHASE_DEVELOPMENT_SERVER
  const backend = process.env.NVFLARE_DASHBOARD_BACKEND_URL?.replace(/\/$/, '')
  return {
    ...(development ? {} : { output: 'export' }),
    basePath: '/nvflare-dashboard',
    agentRules: false, // Keep this repository's agent instructions under contributor control.
    images: { unoptimized: true },
    productionBrowserSourceMaps: true,
    reactStrictMode: true,
    generateBuildId: async () => process.env.NVFLARE_DASHBOARD_BUILD_ID || null,
    ...(development && backend ? {
      async rewrites() {
        return [{ source: '/api/v1/:path*', destination: `${backend}/nvflare-dashboard/api/v1/:path*` }]
      },
    } : {}),
  }
}
