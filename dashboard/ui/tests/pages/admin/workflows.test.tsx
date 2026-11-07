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
import userEvent from '@testing-library/user-event'
import { ReactNode } from 'react'
import { GridColDef, GridRenderCellParams, GridRowSelectionModel } from '@mui/x-data-grid'
import ProjectAdminDashboard from '../../../src/pages/project-admin-dashboard'
import SiteDashboard from '../../../src/pages/site-dashboard'
import { APICall } from '../../../src/apis'
import { ClientFromSchema, User } from '../../../src/types'

// Exercise the page's API/state behavior independently of DataGrid virtualization.
jest.mock('../../../src/components/Layout', () => ({
  __esModule: true,
  default: ({ children }: { children: ReactNode }) => <main>{children}</main>,
}))

jest.mock('../../../src/components/shared', () => ({
  Block: ({ children }: { children: ReactNode }) => <section>{children}</section>,
  Table: ({ columns, data, selectionModel, onSelectionModelChange }: {
    columns: GridColDef[]
    data: Record<string, unknown>[]
    selectionModel: GridRowSelectionModel
    onSelectionModelChange: (selection: GridRowSelectionModel) => void
  }) => (
    <table><tbody>{data.map(row => <tr key={String(row.id)} data-testid={`row-${row.id}`}>
      <td><input
        type="checkbox" aria-label={`Select ${row.email || row.name}`}
        checked={selectionModel.ids.has(String(row.id))}
        onChange={event => {
          const ids = new Set(selectionModel.ids)
          if (event.target.checked) ids.add(String(row.id))
          else ids.delete(String(row.id))
          onSelectionModelChange({ type: 'include', ids })
        }}
      /></td>
      {columns.map(column => <td key={column.field}>
        {column.renderCell
          ? column.renderCell({ row, value: row[column.field], id: String(row.id) } as GridRenderCellParams)
          : row[column.field] as ReactNode}
      </td>)}
    </tr>)}</tbody></table>
  ),
}))

jest.mock('../../../src/apis', () => ({
  APICall: {
    getUsers: jest.fn(), patchUser: jest.fn(), deleteUser: jest.fn(),
    getClients: jest.fn(), patchClient: jest.fn(), deleteClient: jest.fn(),
  },
}))

const api = jest.mocked(APICall)
const users: User[] = [
  { id: 11, name: 'Owner', email: 'owner@example.com', organization: 'Workshop_org', role: 'org_admin', approval_state: 0, created_at: '2026-01-01' },
  { id: 12, name: 'Member', email: 'member@example.com', organization: 'Other_org', role: 'member', approval_state: 0, created_at: '2026-01-01' },
]
const site: ClientFromSchema = {
  id: 31, name: 'workshop-site', organization: 'Workshop_org', approval_state: 0,
  capacity: { num_of_gpus: 4, mem_per_gpu_in_GiB: 16 },
}

beforeEach(() => {
  jest.clearAllMocks()
  api.getUsers.mockResolvedValue({ status: 'ok', user_list: users })
  api.getClients.mockResolvedValue({ status: 'ok', client_list: [site] })
})

test('bulk approval sends reviewed identities and retains out-of-order successful responses', async () => {
  const user = userEvent.setup()
  const pending: ((value: { status: string; user: User }) => void)[] = []
  api.patchUser.mockImplementation(() => new Promise(resolve => pending.push(resolve)))
  render(<ProjectAdminDashboard />)
  expect(await screen.findByText('Org Admin (requested)')).toBeInTheDocument()
  await user.click(screen.getByLabelText('Select owner@example.com'))
  await user.click(screen.getByLabelText('Select member@example.com'))
  await user.click(screen.getByRole('button', { name: 'Approve Selected' }))
  expect(api.patchUser).toHaveBeenCalledWith(11, {
    approval_state: 100, reviewed_role: 'org_admin', reviewed_organization: 'Workshop_org',
  })
  expect(api.patchUser).toHaveBeenCalledWith(12, {
    approval_state: 100, reviewed_role: 'member', reviewed_organization: 'Other_org',
  })
  await act(async () => { pending[1]({ status: 'ok', user: { ...users[1], approval_state: 100 } }) })
  await act(async () => { pending[0]({ status: 'ok', user: { ...users[0], approval_state: 100 } }) })
  expect(within(screen.getByTestId('row-11')).getByText('Approved (D/L: 0)')).toBeInTheDocument()
  expect(within(screen.getByTestId('row-12')).getByText('Approved (D/L: 0)')).toBeInTheDocument()
  expect(screen.queryByText('Org Admin (requested)')).not.toBeInTheDocument()
})

test('a failed bulk approval keeps that registration pending without losing another success', async () => {
  const user = userEvent.setup()
  api.patchUser.mockImplementation(async id => {
    if (id === 12) throw { response: { status: 409, data: { message: 'Review the current registration again.' } } }
    return { status: 'ok', user: { ...users[0], approval_state: 100 } }
  })
  render(<ProjectAdminDashboard />)
  await user.click(await screen.findByLabelText('Select owner@example.com'))
  await user.click(screen.getByLabelText('Select member@example.com'))
  await user.click(screen.getByRole('button', { name: 'Approve Selected' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Review the current registration again.')
  expect(within(screen.getByTestId('row-11')).getByText('Approved (D/L: 0)')).toBeInTheDocument()
  expect(within(screen.getByTestId('row-12')).getByText('Pending')).toBeInTheDocument()
  expect(screen.getByLabelText('Select member@example.com')).toBeChecked()
})

test('rejected profile edits retain entered values and the saved registration', async () => {
  const user = userEvent.setup()
  api.patchUser.mockRejectedValue({ response: { status: 400, data: { message: 'Organization is invalid.' } } })
  render(<ProjectAdminDashboard />)
  await user.click(await screen.findByRole('button', { name: 'Edit organization for owner@example.com' }))
  const organization = screen.getByRole('textbox', { name: 'Organization' })
  await user.clear(organization)
  await user.type(organization, 'Bad org')
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Organization is invalid.')
  expect(organization).toHaveValue('Bad org')
  expect(within(screen.getByTestId('row-11')).getByText('Workshop_org')).toBeInTheDocument()
})

test('client sites can be approved, denied, and deleted with confirmation', async () => {
  const user = userEvent.setup()
  api.patchClient.mockImplementation(async (_id, patch) => ({ status: 'ok', client: { ...site, ...patch } }))
  api.deleteClient.mockResolvedValue({ status: 'ok' })
  render(<SiteDashboard />)
  await user.click(await screen.findByLabelText('Select workshop-site'))
  await user.click(screen.getByRole('button', { name: 'Approve Selected' }))
  expect(api.patchClient).toHaveBeenCalledWith(31, { approval_state: 100 })
  await waitFor(() => expect(within(screen.getByTestId('row-31')).getByText('Approved')).toBeInTheDocument())
  await user.click(screen.getByLabelText('Select workshop-site'))
  await user.click(screen.getByRole('button', { name: 'Deny Selected' }))
  await waitFor(() => expect(within(screen.getByTestId('row-31')).getByText('Denied')).toBeInTheDocument())
  await user.click(screen.getByLabelText('Select workshop-site'))
  await user.click(screen.getByRole('button', { name: 'Delete Selected' }))
  expect(api.deleteClient).not.toHaveBeenCalled()
  await user.click(screen.getByRole('button', { name: 'Delete Sites' }))
  await waitFor(() => expect(screen.queryByTestId('row-31')).not.toBeInTheDocument())
  expect(api.deleteClient).toHaveBeenCalledWith(31)
})

test('site rename addresses the correct record and permits correction after a conflict', async () => {
  const user = userEvent.setup()
  api.patchClient.mockRejectedValueOnce({ response: { status: 409, data: { message: 'This site name already exists.' } } })
    .mockResolvedValueOnce({ status: 'ok', client: { ...site, name: 'corrected-site' } })
  render(<SiteDashboard />)
  await user.click(await screen.findByRole('button', { name: 'Edit name for workshop-site' }))
  const name = screen.getByRole('textbox', { name: 'Site Name' })
  fireEvent.change(name, { target: { value: 'duplicate-site' } })
  await user.click(screen.getByRole('button', { name: 'Save' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('This site name already exists.')
  expect(name).toHaveValue('duplicate-site')
  expect(api.patchClient).toHaveBeenCalledWith(31, { name: 'duplicate-site' })
  fireEvent.change(name, { target: { value: 'corrected-site' } })
  await user.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  expect(within(screen.getByTestId('row-31')).getByText('corrected-site')).toBeInTheDocument()
})
