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

import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useRouter } from 'next/router'
import ProjectHome from '../../../src/pages/index'
import Downloads from '../../../src/pages/downloads'
import ServerConfiguration from '../../../src/pages/server-config'
import ProjectConfiguration from '../../../src/pages/project-configuration'
import { APICall } from '../../../src/apis'
import { getSession, setSession } from '../../../src/sess/session-manager'

jest.mock('../../../src/apis', () => ({ APICall: {
  getProject: jest.fn(), getClients: jest.fn(), getUser: jest.fn(), patchProject: jest.fn(), login: jest.fn(),
  getUserStartupKit: jest.fn(), getClientStartupKit: jest.fn(), getServerStartupKit: jest.fn(),
} }))
jest.mock('components/Layout', () => ({ __esModule: true, default: ({ children }: { children: React.ReactNode }) => <main>{children}</main> }))
jest.mock('components/shared/Table', () => ({ __esModule: true, default: () => <div>Client sites table</div> }))
const api = jest.mocked(APICall)
const project = { short_name: 'workshop', title: 'Workshop', description: '', app_location: '', public: true, frozen: false, server1: 'server.test' }
const user = { id: 1, name: 'Owner', email: 'owner@example.com', role: 'org_admin', organization: 'OrgA', approval_state: 100, created_at: '' }
const site = { id: 10, name: 'site-a', organization: 'OrgA', approval_state: 100 }

beforeEach(() => {
  jest.clearAllMocks()
  localStorage.clear()
  setSession(user.email, 'token', 1, 4)
  api.getProject.mockResolvedValue({ status: 'ok', project })
  api.getUser.mockResolvedValue({ status: 'ok', user })
  api.getClients.mockResolvedValue({ status: 'ok', client_list: [site] })
})

test('failed freeze preserves the unfrozen project and permits retry', async () => {
  api.patchProject.mockRejectedValueOnce({ response: { status: 400, data: { message: 'Server name must be a valid hostname.' } } })
    .mockResolvedValueOnce({ status: 'ok', project: { ...project, frozen: true } })
  render(<ProjectHome />)
  const freezeButton = await screen.findByRole('button', { name: 'Freeze Project' })
  await waitFor(() => expect(freezeButton).toBeEnabled())
  fireEvent.click(freezeButton)
  const dialog = screen.getByRole('dialog')
  fireEvent.click(within(dialog).getByRole('button', { name: 'Freeze' }))
  expect(await within(dialog).findByText('Server name must be a valid hostname.')).toBeVisible()
  expect(screen.getByText('Project status: Setup in progress')).toBeVisible()
  fireEvent.click(within(dialog).getByRole('button', { name: 'Freeze' }))
  expect(await screen.findByText('Project status: Frozen')).toBeVisible()
  expect(api.patchProject).toHaveBeenCalledWith({ frozen: true })
})

test('project is not marked frozen until its API request succeeds', async () => {
  let complete!: (value: { status: string; project: typeof project }) => void
  api.patchProject.mockReturnValueOnce(new Promise(resolve => { complete = resolve }))
  render(<ProjectHome />)
  const freezeButton = await screen.findByRole('button', { name: 'Freeze Project' })
  await waitFor(() => expect(freezeButton).toBeEnabled())
  fireEvent.click(freezeButton)
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Freeze' }))
  expect(screen.getByText('Project status: Setup in progress')).toBeVisible()
  await act(async () => complete({ status: 'ok', project: { ...project, frozen: true } }))
  expect(screen.getByText('Project status: Frozen')).toBeVisible()
})

test('Member login preserves the Member role and opens the user dashboard', async () => {
  setSession('', '', -1, 0)
  const push = jest.fn()
  jest.mocked(useRouter).mockReturnValue({ push } as unknown as ReturnType<typeof useRouter>)
  api.login.mockResolvedValue({ status: 'ok', access_token: 'new-token', user: { ...user, role: 'member' } })
  render(<ProjectHome />)
  fireEvent.change(screen.getByLabelText(/^Email/), { target: { value: user.email } })
  fireEvent.change(screen.getByLabelText(/^Password/), { target: { value: 'secret' } })
  fireEvent.click(screen.getByRole('button', { name: 'Log in' }))
  await waitFor(() => expect(push).toHaveBeenCalledWith('/user-dashboard'))
  expect(getSession().user.role).toBe(1)
})

test('download failure has actionable feedback and retry displays its PIN only after success', async () => {
  api.getProject.mockResolvedValue({ status: 'ok', project: { ...project, frozen: true } })
  api.getClientStartupKit.mockRejectedValueOnce({ response: { status: 409, data: { message: 'Project trust material is not ready.' } } }).mockResolvedValueOnce()
  render(<Downloads />)
  const button = await screen.findByRole('button', { name: 'Download Startup Kit' })
  await waitFor(() => expect(button).toBeEnabled())
  fireEvent.click(button)
  expect(await screen.findByRole('alert')).toHaveTextContent('Project trust material is not ready.')
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  fireEvent.click(button)
  expect(await screen.findByRole('dialog')).toHaveTextContent('The secure PIN is:')
  expect(api.getClientStartupKit).toHaveBeenLastCalledWith(site.id, 'site-a.zip', expect.stringMatching(/^[A-F0-9]{12}$/))
})

test.each([[false, 100, 100], [true, 0, 100], [true, 100, 0]])('download gating respects freeze=%s user=%s site=%s', async (frozen, approval, siteApproval) => {
  api.getProject.mockResolvedValue({ status: 'ok', project: { ...project, frozen: Boolean(frozen) } })
  api.getUser.mockResolvedValue({ status: 'ok', user: { ...user, approval_state: Number(approval) } })
  api.getClients.mockResolvedValue({ status: 'ok', client_list: [{ ...site, approval_state: Number(siteApproval) }] })
  render(<Downloads />)
  expect(await screen.findByRole('button', { name: 'Download Startup Kit' })).toBeDisabled()
})

test('server validation retains the typed hostname and can be corrected', async () => {
  api.patchProject.mockRejectedValueOnce({ response: { status: 400, data: { message: 'Invalid server name.', errors: { server1: 'Invalid server name.' } } } })
    .mockResolvedValueOnce({ status: 'ok', project: { ...project, server1: 'correct.test' } })
  render(<ServerConfiguration />)
  const input = await screen.findByLabelText(/^Server hostname or IP address/)
  fireEvent.change(input, { target: { value: 'bad host' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(input).toHaveValue('bad host'))
  expect(await screen.findAllByText('Invalid server name.')).toHaveLength(2)
  fireEvent.change(input, { target: { value: 'correct.test' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText('Server settings saved.')).toBeVisible()
})


test('hostname saves leave operator server properties untouched and expose no CC controls', async () => {
  const serverProps = { cc_enabled: true, fed_learn_port: 9002, admin_port: 9003, custom: { enabled: true } }
  api.getProject.mockResolvedValue({ status: 'ok', project: { ...project, server_props: serverProps } })
  api.patchProject.mockResolvedValue({ status: 'ok', project: { ...project, server1: 'new.test', server_props: serverProps } })
  render(<ServerConfiguration />)
  const input = await screen.findByLabelText(/^Server hostname or IP address/)
  expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Target Deployment Environment')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('CPU Family')).not.toBeInTheDocument()
  fireEvent.change(input, { target: { value: 'new.test' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText('Server settings saved.')).toBeVisible()
  expect(api.patchProject).toHaveBeenCalledWith({ server1: 'new.test' })
})


test('publishing the project retains unsaved configuration fields', async () => {
  api.getProject.mockResolvedValue({ status: 'ok', project: { ...project, public: false } })
  api.patchProject.mockResolvedValue({ status: 'ok', project: { ...project, public: true } })
  render(<ProjectConfiguration />)
  const title = await screen.findByLabelText('Title')
  fireEvent.change(title, { target: { value: 'Unsaved workshop title' } })
  fireEvent.click(screen.getByRole('button', { name: 'Make Project Public' }))
  await screen.findByRole('button', { name: 'Project is Public' })
  expect(title).toHaveValue('Unsaved workshop title')
  expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled()
  expect(api.patchProject).toHaveBeenCalledWith({ public: true })
})
