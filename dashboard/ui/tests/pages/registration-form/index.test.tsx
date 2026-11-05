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

import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { APICall } from 'apis'
import { useRouter } from 'next/router'
import { getSession, setSession } from 'sess/session-manager'
import RegistrationForm from '../../../src/pages/registration-form'

jest.mock('components/NoMenuLayout', () => ({ __esModule: true, default: ({ children }: { children: React.ReactNode }) => <main>{children}</main> }))
jest.mock('sess/session-manager', () => ({ getSession: jest.fn(), setSession: jest.fn() }))
jest.mock('apis', () => ({ APICall: { getUser: jest.fn(), postUser: jest.fn(), login: jest.fn(), patchUser: jest.fn(), getClients: jest.fn(), postClient: jest.fn(), patchClient: jest.fn(), deleteClient: jest.fn() } }))

const account = { id: 2, email: 'participant@example.test', name: 'Participant', organization: '', role: '', approval_state: 0, created_at: '2026-01-01' }
const session = { user: { id: 2, email: account.email, token: 'token', role: -1 }, status: 'authenticated' }

beforeEach(() => {
  jest.resetAllMocks()
  jest.mocked(useRouter).mockReturnValue({ push: jest.fn() } as unknown as ReturnType<typeof useRouter>)
  jest.mocked(getSession).mockReturnValue({ user: { id: -1, email: '', token: '', role: 0 }, status: 'unauthenticated' })
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: account })
  jest.mocked(APICall.postUser).mockResolvedValue({ status: 'ok', user: account })
  jest.mocked(APICall.login).mockResolvedValue({ status: 'ok', user: account, access_token: 'token' })
  jest.mocked(APICall.getClients).mockResolvedValue({ status: 'ok', client_list: [] })
})

async function enterAccount() {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText('Name'), 'Participant')
  await user.type(screen.getByLabelText('Email'), account.email)
  await user.type(screen.getByLabelText('Password'), 'Testing1234!')
  await user.type(screen.getByLabelText('Confirm password'), 'Testing1234!')
  await user.click(screen.getByRole('button', { name: 'Register' }))
}

test('does not advance registration before automatic login finishes', async () => {
  let finishLogin!: (value: Awaited<ReturnType<typeof APICall.login>>) => void
  jest.mocked(APICall.login).mockReturnValue(new Promise(resolve => { finishLogin = resolve }))
  render(<RegistrationForm />)
  await enterAccount()
  await waitFor(() => expect(APICall.login).toHaveBeenCalled())
  expect(screen.queryByLabelText('Organization')).not.toBeInTheDocument()
  expect(setSession).not.toHaveBeenCalled()
  await act(async () => finishLogin({ status: 'ok', user: account, access_token: 'token' }))
  expect(await screen.findByLabelText('Organization')).toBeVisible()
  expect(setSession).toHaveBeenCalledWith(account.email, 'token', 2, -1)
})

test('failed login after account creation explains how to resume', async () => {
  jest.mocked(APICall.login).mockRejectedValue({ response: { status: 503 } })
  render(<RegistrationForm />)
  await enterAccount()
  expect(await screen.findByText('Your account was created. Please log in from the home page to continue registration.')).toBeVisible()
  expect(screen.getByLabelText('Email')).toHaveValue(account.email)
})

test.each([200, 403, 409, 401])('registration rejection %s retains organization and a useful error', async status => {
  const user = userEvent.setup()
  jest.mocked(getSession).mockReturnValue(session)
  const data = { status: 'error', message: 'Review the requested organization with the Project Admin.' }
  if (status === 200) jest.mocked(APICall.patchUser).mockResolvedValue(data)
  else jest.mocked(APICall.patchUser).mockRejectedValue({ response: { status, data } })
  render(<RegistrationForm />)
  const organization = await screen.findByLabelText('Organization')
  await user.type(organization, 'Workshop_org')
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText(status === 401 ? 'Your session has expired. Please log in again.' : data.message)).toBeVisible()
  expect(organization).toHaveValue('Workshop_org')
  expect(organization).toBeEnabled()
  expect(screen.queryByRole('button', { name: 'Add Site' })).not.toBeInTheDocument()
})

test('Member requests keep the Member session role', async () => {
  const user = userEvent.setup()
  jest.mocked(getSession).mockReturnValue(session)
  jest.mocked(APICall.patchUser).mockResolvedValue({ status: 'ok', user: { ...account, organization: 'Workshop_org', role: 'member' } })
  render(<RegistrationForm />)
  await user.type(await screen.findByLabelText('Organization'), 'Workshop_org')
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText('Thank you for completing your registration.')).toBeVisible()
  expect(setSession).toHaveBeenCalledWith(account.email, 'token', 2, 1, 'Workshop_org')
})

test('Org Admin request stays pending and permits own site registration', async () => {
  const user = userEvent.setup()
  jest.mocked(getSession).mockReturnValue(session)
  jest.mocked(APICall.patchUser).mockResolvedValue({ status: 'ok', user: { ...account, organization: 'Workshop_org', role: 'org_admin' } })
  render(<RegistrationForm />)
  await user.type(await screen.findByLabelText('Organization'), 'Workshop_org')
  await user.click(screen.getByRole('combobox', { name: 'Role' }))
  await user.click(await screen.findByRole('option', { name: 'Org Admin' }))
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText('Org Admin request pending approval: register your own client sites here.')).toBeVisible()
  expect(screen.getByRole('button', { name: 'Add Site' })).toBeEnabled()
  expect(screen.getByLabelText('Organization')).toBeDisabled()
  expect(APICall.patchUser).toHaveBeenCalledWith(2, { organization: 'Workshop_org', role: 'org_admin' })
})


test('resumed registrations display and lock their existing organization', async () => {
  const user = userEvent.setup()
  jest.mocked(getSession).mockReturnValue(session)
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: { ...account, organization: 'Existing_org' } })
  jest.mocked(APICall.patchUser).mockResolvedValue({ status: 'ok', user: { ...account, organization: 'Existing_org', role: 'org_admin' } })
  render(<RegistrationForm />)
  const organization = await screen.findByLabelText('Organization')
  expect(organization).toHaveValue('Existing_org')
  expect(organization).toBeDisabled()
  await user.click(screen.getByRole('combobox', { name: 'Role' }))
  await user.click(await screen.findByRole('option', { name: 'Org Admin' }))
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText('Org Admin request pending approval: register your own client sites here.')).toBeVisible()
  expect(APICall.patchUser).toHaveBeenCalledWith(2, { organization: 'Existing_org', role: 'org_admin' })
})

test('resumed registrations preserve an already requested Lead role', async () => {
  const user = userEvent.setup()
  jest.mocked(getSession).mockReturnValue(session)
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: { ...account, role: 'lead' } })
  jest.mocked(APICall.patchUser).mockResolvedValue({ status: 'ok', user: { ...account, organization: 'Workshop_org', role: 'lead' } })
  render(<RegistrationForm />)
  await user.type(await screen.findByLabelText('Organization'), 'Workshop_org')
  expect(screen.getByRole('combobox', { name: 'Role' })).toHaveAttribute('aria-disabled', 'true')
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByText('Thank you for completing your registration.')).toBeVisible()
  expect(APICall.patchUser).toHaveBeenCalledWith(2, { organization: 'Workshop_org', role: 'lead' })
})

test('legacy approved blank profiles require explicit administrator repair', async () => {
  jest.mocked(getSession).mockReturnValue(session)
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: { ...account, approval_state: 100 } })
  render(<RegistrationForm />)
  expect(await screen.findByText('Your account is already approved. Contact the Project Admin to complete or correct its role and organization.')).toBeVisible()
  expect(screen.getByLabelText('Organization')).toBeDisabled()
  expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
  expect(APICall.patchUser).not.toHaveBeenCalled()
})

test('a stale incomplete session reloads the completed profile instead of requesting new permissions', async () => {
  jest.mocked(getSession).mockReturnValue(session)
  jest.mocked(APICall.getUser).mockResolvedValue({ status: 'ok', user: { ...account, organization: 'Reviewed_org', role: 'member', approval_state: 100 } })
  render(<RegistrationForm />)
  await waitFor(() => expect(jest.mocked(useRouter)().push).toHaveBeenCalledWith('/user-dashboard'))
  expect(setSession).toHaveBeenCalledWith(account.email, 'token', 2, 1, 'Reviewed_org')
  expect(APICall.patchUser).not.toHaveBeenCalled()
})
