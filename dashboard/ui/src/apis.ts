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

import axios, { AxiosResponse } from 'axios'
import FileDownload from 'js-file-download'
import siteparams from '../siteparams.json'
import { AxiosDataClientList, AxiosDataProject, AxiosDataUser, AxiosDataUsers, ClientFromSchema, Project, User } from './types'
import { getSession, setSession } from './sess/session-manager'

export type ApiResult = { status: string; message?: string; errors?: Record<string, string> }
export type UserPatch = Partial<User> & { reviewed_role?: string; reviewed_organization?: string }

export const getAuthHeader = () => getSession().user.token

const instance = axios.create({ baseURL: `${siteparams.url_root}/api/v1/` })
instance.interceptors.request.use(config => {
  const token = getAuthHeader()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})
instance.interceptors.response.use(response => response, error => {
  if (error.response?.status === 401 || error.response?.status === 422) {
    setSession('', '', -1, 0, '', error.config?.url !== 'login')
  }
  return Promise.reject(error)
})

const responseBody = <T,>(response: AxiosResponse<T>): T => response.data

async function download(url: string, fallbackFilename: string, pin: string): Promise<void> {
  const response = await instance.post<Blob>(url, { pin }, { responseType: 'blob' })
  if (String(response.headers['content-type']).includes('application/json')) {
    let message = 'The server could not prepare the startup kit.'
    try {
      const body = JSON.parse(await response.data.text()) as ApiResult
      message = body.message || message
    } catch { /* Use the readable fallback for malformed responses. */ }
    throw new Error(message)
  }
  const disposition = String(response.headers['content-disposition'] || '')
  const filename = /filename="([^"]+)"/.exec(disposition)?.[1] || fallbackFilename
  FileDownload(response.data, filename)
}

export const APICall = {
  login: (data: { email: string; password: string }): Promise<ApiResult & { user: User; access_token: string }> => instance.post('login', data).then(responseBody),
  getUsers: (): Promise<AxiosDataUsers> => instance.get('users').then(responseBody),
  getUser: (id: number): Promise<AxiosDataUser> => instance.get(`users/${id}`).then(responseBody),
  getUserStartupKit: (id: number, filename: string, pin: string) => download(`users/${id}/blob`, filename, pin),
  getClientStartupKit: (id: number, filename: string, pin: string) => download(`clients/${id}/blob`, filename, pin),
  getServerStartupKit: (filename: string, pin: string) => download('servers/1/blob', filename, pin),
  getClients: (): Promise<AxiosDataClientList> => instance.get('clients').then(responseBody),
  getProject: (): Promise<AxiosDataProject> => instance.get('project').then(responseBody),
  postUser: (data: Partial<User>): Promise<AxiosDataUser> => instance.post('users', data).then(responseBody),
  patchUser: (id: number, data: UserPatch): Promise<AxiosDataUser> => instance.patch(`users/${id}`, data).then(responseBody),
  deleteUser: (id: number): Promise<ApiResult> => instance.delete(`users/${id}`).then(responseBody),
  postClient: (data: Partial<ClientFromSchema>): Promise<ApiResult & { client: ClientFromSchema }> => instance.post('clients', data).then(responseBody),
  patchClient: (id: number, data: Partial<ClientFromSchema>): Promise<ApiResult & { client: ClientFromSchema }> => instance.patch(`clients/${id}`, data).then(responseBody),
  deleteClient: (id: number): Promise<ApiResult> => instance.delete(`clients/${id}`).then(responseBody),
  patchProject: (data: Partial<Project>): Promise<AxiosDataProject> => instance.patch('project', data).then(responseBody),
}
