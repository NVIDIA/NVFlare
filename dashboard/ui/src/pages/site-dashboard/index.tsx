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
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  TextField,
} from '@mui/material'
import { GridColDef, GridRowSelectionModel } from '@mui/x-data-grid'
import dayjs from 'dayjs'
import Layout from '../../components/Layout'
import { Block, Table } from '../../components/shared'
import { APICall } from '../../apis'
import { getErrorMessage } from '../../api-errors'
import { ClientFromSchema } from '../../types'

type BulkAction = 'approve' | 'deny' | 'delete'
type SiteEdit = { site: ClientFromSchema; field: 'name' | 'organization'; value: string }

const emptySelection = (): GridRowSelectionModel => ({ type: 'include', ids: new Set() })

const SiteDashboard = () => {
  const [sites, setSites] = useState<ClientFromSchema[]>([])
  const [selection, setSelection] = useState<GridRowSelectionModel>(emptySelection)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [edit, setEdit] = useState<SiteEdit | null>(null)
  const [editError, setEditError] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const response = await APICall.getClients()
        if (response.status !== 'ok' || !Array.isArray(response.client_list)) {
          throw new Error(response.message || 'Unable to load registered client sites.')
        }
        if (active) setSites(response.client_list)
      } catch (failure) {
        if (active) setError(getErrorMessage(failure, 'Unable to load registered client sites.'))
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const selectedSites = useMemo(() => sites.filter(site => {
    const selected = selection.ids.has(String(site.id))
    return selection.type === 'include' ? selected : !selected
  }), [selection, sites])

  const mergeSite = (site: ClientFromSchema) => {
    setSites(previous => previous.map(current => current.id === site.id ? site : current))
  }

  const openEdit = (site: ClientFromSchema, field: SiteEdit['field']) => {
    setEditError('')
    setEdit({ site, field, value: site[field] })
  }

  const saveEdit = async () => {
    if (!edit) return
    setBusy(true)
    setEditError('')
    try {
      const response = await APICall.patchClient(edit.site.id, { [edit.field]: edit.value })
      if (response.status !== 'ok' || !response.client) {
        throw new Error(response.message || 'Unable to save the client site.')
      }
      mergeSite(response.client)
      setEdit(null)
    } catch (failure) {
      setEditError(getErrorMessage(failure, 'Unable to save the client site. Please check the entered value.'))
    } finally {
      setBusy(false)
    }
  }

  const runBulkAction = async (action: BulkAction) => {
    const selected = [...selectedSites]
    setBusy(true)
    setError('')
    setConfirmDelete(false)
    const failures: string[] = []
    const failedIds = new Set<string>()
    await Promise.all(selected.map(async site => {
      try {
        if (action === 'delete') {
          const response = await APICall.deleteClient(site.id)
          if (response.status !== 'ok') throw new Error(response.message || 'Unable to delete this client site.')
          setSites(previous => previous.filter(current => current.id !== site.id))
        } else {
          const response = await APICall.patchClient(site.id, { approval_state: action === 'approve' ? 100 : -1 })
          if (response.status !== 'ok' || !response.client) {
            throw new Error(response.message || 'Unable to update the client site approval.')
          }
          mergeSite(response.client)
        }
      } catch (failure) {
        failedIds.add(String(site.id))
        failures.push(`${site.name}: ${getErrorMessage(failure, 'The operation failed. Please try again.')}`)
      }
    }))
    setSelection({ type: 'include', ids: failedIds })
    setError(failures.join('\n'))
    setBusy(false)
  }

  const columns: GridColDef<ClientFromSchema>[] = [
    {
      field: 'name', headerName: 'Name', minWidth: 200, flex: 1,
      renderCell: ({ row }) => <Button variant="text" onClick={() => openEdit(row, 'name')} aria-label={`Edit name for ${row.name}`}>{row.name || '[none]'}</Button>,
    },
    {
      field: 'organization', headerName: 'Organization', minWidth: 180, flex: 1,
      renderCell: ({ row }) => <Button variant="text" onClick={() => openEdit(row, 'organization')} aria-label={`Edit organization for ${row.name}`}>{row.organization || '[none]'}</Button>,
    },
    {
      field: 'capacity', headerName: 'Capacity', minWidth: 240,
      renderCell: ({ row }) => `GPUs: ${row.capacity?.num_of_gpus ?? '—'}; Memory/GPU: ${row.capacity?.mem_per_gpu_in_GiB ?? '—'} GiB`,
    },
    {
      field: 'created_at', headerName: 'Created At', minWidth: 170,
      renderCell: ({ value }) => value ? dayjs(value).format('MM/DD/YYYY HH:mm:ss') : '',
    },
    {
      field: 'approval_state', headerName: 'Approval', minWidth: 140,
      renderCell: ({ row }) => row.approval_state < 0 ? 'Denied' : row.approval_state >= 100 ? 'Approved' : 'Pending',
    },
    { field: 'download_count', headerName: 'Download Count', minWidth: 150, renderCell: ({ value }) => value || 0 },
  ]

  return (
    <Layout title="Client Sites Dashboard">
      <Block title="Client Sites">
        {error && <Alert severity="error" sx={{ mb: 2, whiteSpace: 'pre-line' }}>{error}</Alert>}
        <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
          <Button disabled={busy || selectedSites.length === 0} variant="contained" onClick={() => void runBulkAction('approve')}>Approve Selected</Button>
          <Button disabled={busy || selectedSites.length === 0} onClick={() => void runBulkAction('deny')}>Deny Selected</Button>
          <Button disabled={busy || selectedSites.length === 0} color="error" onClick={() => setConfirmDelete(true)}>Delete Selected</Button>
        </Stack>
        <Table<ClientFromSchema>
          columns={columns} data={sites} loading={loading} checkboxSelection
          getRowId={site => String(site.id)} selectionModel={selection}
          onSelectionModelChange={setSelection} disableRowSelectionOnClick
        />
      </Block>
      <Dialog open={edit !== null} onClose={() => { if (!busy) setEdit(null) }} fullWidth maxWidth="sm">
        <DialogTitle>Edit client {edit?.field}</DialogTitle>
        <DialogContent>
          {editError && <Alert severity="error" sx={{ mb: 2 }}>{editError}</Alert>}
          {edit && <TextField
            fullWidth autoFocus margin="dense" label={edit.field === 'name' ? 'Site Name' : 'Organization'}
            value={edit.value} disabled={busy} onChange={event => setEdit({ ...edit, value: event.target.value })}
          />}
        </DialogContent>
        <DialogActions>
          <Button disabled={busy} onClick={() => setEdit(null)}>Cancel</Button>
          <Button disabled={busy || !edit?.value.trim()} onClick={() => void saveEdit()}>Save</Button>
        </DialogActions>
      </Dialog>
      <Dialog open={confirmDelete} onClose={() => setConfirmDelete(false)}>
        <DialogTitle>Delete selected client sites?</DialogTitle>
        <DialogContent>Delete {selectedSites.length} selected client-site registration(s)?</DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
          <Button color="error" onClick={() => void runBulkAction('delete')}>Delete Sites</Button>
        </DialogActions>
      </Dialog>
    </Layout>
  )
}

export default SiteDashboard
