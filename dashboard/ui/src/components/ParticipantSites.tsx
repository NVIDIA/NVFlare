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

import { useState } from 'react'
import { Alert, Button, Dialog, DialogContent, DialogTitle, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material'
import { getErrorMessage } from 'api-errors'
import { newSite } from 'participant-sites'
import ParticipantSiteEditor from './ParticipantSiteEditor'
import type { SiteView } from '../types'

interface ParticipantSitesProps {
  sites: SiteView[]
  organization: string
  loading?: boolean
  onSave: (site: SiteView) => Promise<void>
  onDelete: (site: SiteView) => Promise<void>
}

export default function ParticipantSites({ sites, organization, loading = false, onSave, onDelete }: ParticipantSitesProps) {
  const [editing, setEditing] = useState<SiteView | null>(null)
  const [error, setError] = useState('')
  const [deleting, setDeleting] = useState<number | null>(null)

  async function deleteSite(site: SiteView) {
    setError('')
    setDeleting(site.id)
    try {
      await onDelete(site)
    } catch (failure) {
      setError(getErrorMessage(failure, 'Unable to delete the client site. Please reload and try again.'))
    } finally {
      setDeleting(null)
    }
  }

  return (
    <Stack spacing={2}>
      <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography component="h2" variant="h6">Client Sites</Typography>
        <Button variant="outlined" disabled={loading} onClick={() => setEditing(newSite(organization))}>Add Site</Button>
      </Stack>
      {error && <Alert severity="error">{error}</Alert>}
      <TableContainer component={Paper} variant="outlined">
        <Table aria-label="Your client sites">
          <TableHead><TableRow>
            <TableCell>Name</TableCell><TableCell>GPUs</TableCell><TableCell>Memory per GPU (GiB)</TableCell><TableCell>Status</TableCell><TableCell>Actions</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {sites.map(site => (
              <TableRow key={site.id}>
                <TableCell><Button onClick={() => setEditing({ ...site })}>{site.name}</Button></TableCell>
                <TableCell>{site.num_of_gpus}</TableCell>
                <TableCell>{site.mem_per_gpu_in_GiB}</TableCell>
                <TableCell>{Number(site.approval_state) >= 100 ? 'Approved' : Number(site.approval_state) < 0 ? 'Denied' : 'Pending'}</TableCell>
                <TableCell><Button color="error" aria-label={`Delete ${site.name}`} disabled={Number(site.approval_state) >= 100 || deleting !== null} onClick={() => void deleteSite(site)}>Delete</Button></TableCell>
              </TableRow>
            ))}
            {sites.length === 0 && <TableRow><TableCell colSpan={5}>{loading ? 'Loading client sites…' : 'No client sites registered.'}</TableCell></TableRow>}
          </TableBody>
        </Table>
      </TableContainer>
      <Dialog open={editing !== null} onClose={(_event, reason) => { if (reason !== 'backdropClick') setEditing(null) }} fullWidth maxWidth="sm">
        <DialogTitle>Edit Site Details</DialogTitle>
        <DialogContent>
          {editing && <ParticipantSiteEditor site={editing} onClose={() => setEditing(null)} onSave={onSave} />}
        </DialogContent>
      </Dialog>
    </Stack>
  )
}
