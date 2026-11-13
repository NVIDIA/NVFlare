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

import type { ClientFromSchema, SiteView } from './types'

export function toSiteView(client: ClientFromSchema): SiteView {
  return {
    id: client.id,
    uid: String(client.id),
    name: client.name,
    org: client.organization,
    approval_state: client.approval_state,
    num_of_gpus: client.capacity?.num_of_gpus ?? 0,
    mem_per_gpu_in_GiB: client.capacity?.mem_per_gpu_in_GiB ?? 0,
  }
}

export function siteRequest(site: SiteView, organization: string) {
  return {
    name: site.name,
    organization,
    capacity: {
      num_of_gpus: Number(site.num_of_gpus),
      mem_per_gpu_in_GiB: Number(site.mem_per_gpu_in_GiB),
    },
  }
}

export function newSite(organization: string): SiteView {
  return {
    id: -1,
    uid: 'new-site',
    name: '',
    org: organization,
    approval_state: 0,
    num_of_gpus: 4,
    mem_per_gpu_in_GiB: 16,
  }
}
