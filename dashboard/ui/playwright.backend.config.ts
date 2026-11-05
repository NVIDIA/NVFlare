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

import { defineConfig } from '@playwright/test'

export default defineConfig({
  outputDir: './test-results/backend',
  testDir: './tests/backend',
  workers: 1,
  timeout: 60000,
  use: { baseURL: 'http://127.0.0.1:3110', channel: process.env.NVFLARE_TEST_BROWSER_CHANNEL || undefined,
    viewport: { width: 1600, height: 1000 }, trace: 'retain-on-failure' },
  webServer: { command: 'python3 tests/serve_backend.py', url: 'http://127.0.0.1:3110/nvflare-dashboard/', timeout: 30000 },
})
