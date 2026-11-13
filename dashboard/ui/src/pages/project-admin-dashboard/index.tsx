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

import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { GridColDef, GridRowSelectionModel } from '@mui/x-data-grid'
import dayjs from 'dayjs'
import Layout from '../../components/Layout'
import { Block, Table } from '../../components/shared'
import { APICall } from '../../apis'
import { getErrorMessage } from '../../api-errors'
import { User } from '../../types'

type EditableField = 'name' | 'organization' | 'role'
type BulkAction = 'approve' | 'deny' | 'delete'
type UserEdit = { user: User; field: EditableField; value: string }

const emptySelection = (): GridRowSelectionModel => ({ type: 'include', ids: new Set() })
const roleNames: Record<string, string> = {
  project_admin: 'Project Admin', org_admin: 'Org Admin', lead: 'Lead', member: 'Member',
}

function requestedRole(user: User) {
  const role = roleNames[user.role] || user.role || '[none]'
  return user.role === 'org_admin' && Number(user.approval_state) >= 0 && Number(user.approval_state) < 100
    ? `${role} (requested)` : role
}

const ProjectAdminDashboard = () => {
  const [users, setUsers] = useState<User[]>([])
  const [selection, setSelection] = useState<GridRowSelectionModel>(emptySelection)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [edit, setEdit] = useState<UserEdit | null>(null)
  const [editError, setEditError] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const response = await APICall.getUsers()
        if (response.status !== 'ok' || !Array.isArray(response.user_list)) {
          throw new Error(response.message || 'Unable to load registered users.')
        }
        if (active) setUsers(response.user_list)
      } catch (failure) {
        if (active) setError(getErrorMessage(failure, 'Unable to load registered users.'))
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const selectedUsers = useMemo(() => users.filter(user => {
    const selected = selection.ids.has(String(user.id))
    return selection.type === 'include' ? selected : !selected
  }), [selection, users])

  const openEdit = (user: User, field: EditableField) => {
    setEditError('')
    setEdit({ user, field, value: user[field] })
  }

  const mergeUser = (user: User) => {
    setUsers(previous => previous.map(current => current.id === user.id ? user : current))
  }

  const saveEdit = async () => {
    if (!edit) return
    setBusy(true)
    setEditError('')
    try {
      const response = await APICall.patchUser(edit.user.id, { [edit.field]: edit.value })
      if (response.status !== 'ok' || !response.user) {
        throw new Error(response.message || 'Unable to update this user.')
      }
      mergeUser(response.user)
      setEdit(null)
    } catch (failure) {
      setEditError(getErrorMessage(failure, 'Unable to update this user. Please check the entered value.'))
    } finally {
      setBusy(false)
    }
  }

  const runBulkAction = async (action: BulkAction) => {
    const reviewedUsers = [...selectedUsers]
    setBusy(true)
    setError('')
    setConfirmDelete(false)
    const failures: string[] = []
    const failedIds = new Set<string>()
    await Promise.all(reviewedUsers.map(async user => {
      try {
        if (action === 'delete') {
          const response = await APICall.deleteUser(user.id)
          if (response.status !== 'ok') throw new Error(response.message || 'Unable to delete this user.')
          setUsers(previous => previous.filter(current => current.id !== user.id))
        } else {
          const request = action === 'approve'
            ? { approval_state: 100, reviewed_role: user.role, reviewed_organization: user.organization }
            : { approval_state: -1 }
          const response = await APICall.patchUser(user.id, request)
          if (response.status !== 'ok' || !response.user) {
            throw new Error(response.message || 'Unable to update approval. Reload and review the registration again.')
          }
          mergeUser(response.user)
        }
      } catch (failure) {
        failedIds.add(String(user.id))
        failures.push(`${user.email}: ${getErrorMessage(failure, 'The operation failed. Please try again.')}`)
      }
    }))
    setSelection({ type: 'include', ids: failedIds })
    setError(failures.join('\n'))
    setBusy(false)
  }

  const columns: GridColDef<User>[] = [
    {
      field: 'name', headerName: 'Name', minWidth: 160, flex: 1,
      renderCell: ({ row }) => <Button variant="text" onClick={() => openEdit(row, 'name')} aria-label={`Edit name for ${row.email}`}>{row.name || '[none]'}</Button>,
    },
    { field: 'email', headerName: 'Email', minWidth: 220, flex: 1 },
    {
      field: 'organization', headerName: 'Organization', minWidth: 180, flex: 1,
      renderCell: ({ row }) => <Button variant="text" onClick={() => openEdit(row, 'organization')} aria-label={`Edit organization for ${row.email}`}>{row.organization || '[none]'}</Button>,
    },
    {
      field: 'role', headerName: 'Role', minWidth: 200, flex: 1,
      renderCell: ({ row }) => row.role === 'project_admin' ? requestedRole(row) : (
        <Button variant="text" onClick={() => openEdit(row, 'role')} aria-label={`Edit role for ${row.email}`}>{requestedRole(row)}</Button>
      ),
    },
    {
      field: 'created_at', headerName: 'Created At', minWidth: 170,
      renderCell: ({ value }) => value ? dayjs(value).format('MM/DD/YYYY HH:mm:ss') : '',
    },
    {
      field: 'approval_state', headerName: 'Approval', minWidth: 190,
      renderCell: ({ row }) => Number(row.approval_state) < 0 ? 'Denied' : Number(row.approval_state) >= 100
        ? `Approved (D/L: ${row.download_count || 0})` : 'Pending',
    },
  ]

  return (
    <Layout title="Users Dashboard">
      <Block title="Users">
        <Typography sx={{ mb: 2 }}>Approval authorizes the displayed role and organization. Review both before approving a request.</Typography>
        {error && <Alert severity="error" sx={{ mb: 2, whiteSpace: 'pre-line' }}>{error}</Alert>}
        <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
          <Button disabled={busy || selectedUsers.length === 0} variant="contained" onClick={() => void runBulkAction('approve')}>Approve Selected</Button>
          <Button disabled={busy || selectedUsers.length === 0} onClick={() => void runBulkAction('deny')}>Deny Selected</Button>
          <Button disabled={busy || selectedUsers.length === 0} color="error" onClick={() => setConfirmDelete(true)}>Delete Selected</Button>
        </Stack>
        <Table<User>
          columns={columns} data={users} loading={loading} checkboxSelection
          getRowId={user => String(user.id)} selectionModel={selection}
          onSelectionModelChange={setSelection} disableRowSelectionOnClick
        />
        <Typography variant="body2" sx={{ mt: 2 }}>D/L is the FLARE Console download count.</Typography>
      </Block>
      <Dialog open={edit !== null} onClose={() => { if (!busy) setEdit(null) }} fullWidth maxWidth="sm">
        <DialogTitle>Edit {edit?.field === 'organization' ? 'organization' : edit?.field}</DialogTitle>
        <DialogContent>
          {editError && <Alert severity="error" sx={{ mb: 2 }}>{editError}</Alert>}
          {edit && <TextField
            fullWidth autoFocus margin="dense" label={edit.field === 'organization' ? 'Organization' : edit.field === 'role' ? 'Role' : 'Name'}
            select={edit.field === 'role'} value={edit.value} disabled={busy}
            onChange={event => setEdit({ ...edit, value: event.target.value })}
          >
            {edit.field === 'role' && ['member', 'lead', 'org_admin'].map(role => <MenuItem key={role} value={role}>{roleNames[role]}</MenuItem>)}
            {edit.field === 'role' && edit.value && !['member', 'lead', 'org_admin'].includes(edit.value) && <MenuItem value={edit.value}>{edit.value}</MenuItem>}
          </TextField>}
        </DialogContent>
        <DialogActions>
          <Button disabled={busy} onClick={() => setEdit(null)}>Cancel</Button>
          <Button disabled={busy || !edit?.value.trim()} onClick={() => void saveEdit()}>Save</Button>
        </DialogActions>
      </Dialog>
      <Dialog open={confirmDelete} onClose={() => setConfirmDelete(false)}>
        <DialogTitle>Delete selected users?</DialogTitle>
        <DialogContent><Box>Delete {selectedUsers.length} selected user registration(s) and their client sites?</Box></DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
          <Button color="error" onClick={() => void runBulkAction('delete')}>Delete Users</Button>
        </DialogActions>
      </Dialog>
    </Layout>
  )
}

export default ProjectAdminDashboard
