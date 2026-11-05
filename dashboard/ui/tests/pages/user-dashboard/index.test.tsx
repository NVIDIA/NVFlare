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

import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { APICall } from 'apis'
import UserDashboard from '../../../src/pages/user-dashboard'

jest.mock('components/Layout', () => ({ __esModule: true, default: ({ children }: { children: React.ReactNode }) => <main>{children}</main> }))
jest.mock('sess/session-manager', () => ({ getSession: () => ({ user: { id: 1, role: 1, token: 'token' } }) }))
jest.mock('apis', () => ({ APICall: { getUser: jest.fn(), getClients: jest.fn(), postClient: jest.fn(), patchClient: jest.fn(), deleteClient: jest.fn(), patchUser: jest.fn() } }))

const pendingUser = { id: 1, name: 'Site Owner', email: 'owner@example.test', organization: 'Workshop_org', role: 'org_admin', approval_state: 0, created_at: '2026-01-01' }
const site = { id: 8, name: 'workshop-site', organization: 'Workshop_org', approval_state: 0, capacity: { num_of_gpus: 4, mem_per_gpu_in_GiB: 16 } }

beforeEach(() => {
  jest.resetAllMocks()
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: pendingUser })
  jest.mocked(APICall.getClients).mockResolvedValue({ status: 'ok', client_list: [site] })
})

test('pending database role permits own sites despite a stale session role', async () => {
  render(<UserDashboard />)
  expect(await screen.findByText('Requested role: Org Admin')).toBeVisible()
  expect(await screen.findByRole('button', { name: 'workshop-site' })).toBeVisible()
  expect(screen.getByText(/requested role and organization are awaiting approval/)).toBeVisible()
  expect(screen.queryByLabelText('CC Enabled')).not.toBeInTheDocument()
})

test('site policy failure retains the edited name and permits retry', async () => {
  const user = userEvent.setup()
  jest.mocked(APICall.patchClient).mockRejectedValueOnce({ response: { status: 403, data: { message: 'Reload the current site approval.' } } })
  render(<UserDashboard />)
  await user.click(await screen.findByRole('button', { name: 'workshop-site' }))
  const dialog = screen.getByRole('dialog')
  const name = within(dialog).getByLabelText('Site Name')
  await user.clear(name)
  await user.type(name, 'edited-site')
  await user.click(within(dialog).getByRole('button', { name: 'Save' }))
  expect(await within(dialog).findByText('Reload the current site approval.')).toBeVisible()
  expect(name).toHaveValue('edited-site')
  expect(name).toBeEnabled()
  await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  expect(screen.getByRole('button', { name: 'workshop-site' })).toBeVisible()
})

test('creating a site sends numeric capacity without client properties', async () => {
  const user = userEvent.setup()
  jest.mocked(APICall.postClient).mockResolvedValue({ status: 'ok', client: { ...site, id: 9, name: 'cpu-site', capacity: { num_of_gpus: 0, mem_per_gpu_in_GiB: 0 } } })
  render(<UserDashboard />)
  await user.click(await screen.findByRole('button', { name: 'Add Site' }))
  const dialog = screen.getByRole('dialog')
  await user.type(within(dialog).getByLabelText('Site Name'), 'cpu-site')
  const gpus = within(dialog).getByLabelText('Number of GPUs')
  const memory = within(dialog).getByLabelText('Memory per GPU (GiB)')
  await user.clear(gpus)
  await user.type(gpus, '0')
  await user.clear(memory)
  await user.type(memory, '0')
  await user.click(within(dialog).getByRole('button', { name: 'Save' }))
  expect(await screen.findByRole('button', { name: 'cpu-site' })).toBeVisible()
  expect(APICall.postClient).toHaveBeenCalledWith({ name: 'cpu-site', organization: 'Workshop_org', capacity: { num_of_gpus: 0, mem_per_gpu_in_GiB: 0 } })
})

test('canceling a new site then editing an existing site uses PATCH', async () => {
  const user = userEvent.setup()
  jest.mocked(APICall.patchClient).mockResolvedValue({ status: 'ok', client: { ...site, name: 'renamed-site' } })
  render(<UserDashboard />)
  await user.click(await screen.findByRole('button', { name: 'Add Site' }))
  await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  await user.click(screen.getByRole('button', { name: 'workshop-site' }))
  const dialog = screen.getByRole('dialog')
  const name = within(dialog).getByLabelText('Site Name')
  await user.clear(name)
  await user.type(name, 'renamed-site')
  await user.click(within(dialog).getByRole('button', { name: 'Save' }))
  expect(await screen.findByRole('button', { name: 'renamed-site' })).toBeVisible()
  expect(APICall.patchClient).toHaveBeenCalledWith(8, expect.objectContaining({ name: 'renamed-site' }))
  expect(APICall.postClient).not.toHaveBeenCalled()
})

test('approved sites cannot be renamed or deleted by their owner', async () => {
  const user = userEvent.setup()
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: { ...pendingUser, approval_state: 100 } })
  jest.mocked(APICall.getClients).mockResolvedValue({ status: 'ok', client_list: [{ ...site, approval_state: 100 }] })
  render(<UserDashboard />)
  const edit = await screen.findByRole('button', { name: 'workshop-site' })
  expect(screen.getByRole('button', { name: 'Delete workshop-site' })).toBeDisabled()
  await user.click(edit)
  expect(within(screen.getByRole('dialog')).getByLabelText('Site Name')).toBeDisabled()
})

test('members do not request client-site data', async () => {
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: { ...pendingUser, role: 'member' } })
  render(<UserDashboard />)
  expect(await screen.findByText('Requested role: Member')).toBeVisible()
  expect(screen.queryByRole('button', { name: 'Add Site' })).not.toBeInTheDocument()
  expect(APICall.getClients).not.toHaveBeenCalled()
})

test('a rejected password change stays editable without changing identity fields', async () => {
  const user = userEvent.setup()
  jest.mocked(APICall.patchUser).mockRejectedValue({ response: { status: 403, data: { message: 'Contact the Project Admin before changing this account.' } } })
  render(<UserDashboard />)
  await user.click(await screen.findByRole('button', { name: 'Edit My Profile' }))
  const dialog = screen.getByRole('dialog')
  await user.type(within(dialog).getByLabelText('New password'), 'NewPassword123!')
  await user.type(within(dialog).getByLabelText('Confirm new password'), 'NewPassword123!')
  await user.click(within(dialog).getByRole('button', { name: 'Save' }))
  expect(await within(dialog).findByText('Contact the Project Admin before changing this account.')).toBeVisible()
  expect(within(dialog).getByLabelText('New password')).toHaveValue('NewPassword123!')
  expect(APICall.patchUser).toHaveBeenCalledWith(1, { password: 'NewPassword123!' })
})
