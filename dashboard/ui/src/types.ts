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

export type SelectOption = {
  label: string;
  value: string;
}

export type User = {
  email: string
  name: string
  id: number
  description?: string
  organization: string
  organization_id?: number
  role: string
  role_id?: number
  password?: string
  confirm_password?: string
  password_hash?: string
  site_names?: string[]
  sites?: Site[]
  created_at: Date | string
  approval_state: string | number
  download_count?: number
  extras?: string
  updated_at?: string
}

export type AdminClientUser = {
  email: string
  first_name: string
  last_name: string
  id: number
  org: string
  role: string
  approved: boolean
}

export type Org = {
  name: string
  id: number
}

export type Site = {
  name: string
  uid: string
  id: number
  num_of_gpus?: number
  mem_per_gpu_in_GiB?: number
  org: string
  approval_state?: number
  download_count?: number
  editing?: boolean
  props?: {
    has_nvidia_cc_enabled_gpu?: boolean
    cpu_family?: string
    cpu_model?: string
    cpu_threads?: number
    cc_enabled?: boolean
    deployment_environment?: string
  }
}

export type SiteView = {
  name: string
  uid: string
  id: number
  num_of_gpus?: number
  mem_per_gpu_in_GiB?: number
  org: string
  approval_state?: number
  download_count?: number
  editing?: boolean
  props?: {
    has_nvidia_cc_enabled_gpu?: boolean
    cpu_family?: string | SelectOption
    cpu_model?: string
    cpu_threads?: number
    cc_enabled?: boolean
    deployment_environment?: string | SelectOption
  }
}

export type SiteDetail = {
  name: string
  id: number
  capacity?: Capacity
  organization?: string
  approval_state?: number
  download_count?: number
  editing?: boolean
}

export type Capacity = {
  num_of_gpus: number
  mem_per_gpu_in_GiB: number
}

export type ClientFromSchema = {
  name: string
  id: number
  description?: string
  organization: string
  organization_id?: number
  target_server?: string
  capacity?: Capacity
  capacity_id?: number
  approval_state: number
  props?: {
    has_nvidia_cc_enabled_gpu?: boolean
    cpu_family?: string
    cpu_model?: string
    cpu_threads?: number
    cc_enabled?: boolean
    deployment_environment?: string
  }
  extras?: string
  download_count?: number
  created_at?: string
  updated_at?: string
}

export type RoleItem = {
  label: string
  value: string
}

export type Project = {
  short_name: string
  title?: string
  description: string
  app_location: string
  num_clients?: number
  num_orgs?: number
  num_users?: number
  starting_date?: string
  end_date?: string
  server1?: string
  server_extras?: string
  public?: boolean
  frozen?: boolean
  project_props?: {
    cc_enabled?: boolean
  }
  server_props?: {
    cc_enabled?: boolean
    cpu_family?: string
    deployment_environment?: string
  }
}

export type ProjectView = {
  short_name: string
  title?: string
  description: string
  app_location: string
  num_clients?: number
  num_orgs?: number
  num_users?: number
  starting_date?: string
  end_date?: string
  server1?: string
  server_extras?: string
  public?: boolean
  frozen?: boolean
  project_props?: {
    cc_enabled?: boolean
  }
  server_props: {
    cc_enabled: boolean
    deployment_environment?: string | SelectOption
    cpu_family?: string | SelectOption
  }
}

export interface Siteparams {
  projectname: string
  demo: boolean
  url_root: string
  arraydata: unknown[]
  default_project_data: ProjectView
}

export interface AxiosDataUsers {
  message?: string
  errors?: Record<string, string>
  status: string,
  user_list: User[]
}

export interface AxiosDataProject {
  message?: string
  errors?: Record<string, string>
  status: string,
  project: Project
}

export interface AxiosDataUser {
  message?: string
  errors?: Record<string, string>
  status: string,
  user?: User
}

export interface AxiosDataClientList {
  message?: string
  errors?: Record<string, string>
  status: string,
  client_list: ClientFromSchema[]
}
