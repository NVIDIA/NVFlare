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

import { Alert, Button, Stack, TextField } from '@mui/material'
import { Form, Formik } from 'formik'
import * as Yup from 'yup'
import { getErrorMessage, getErrorStatus } from 'api-errors'
import type { SiteView } from '../types'

interface ParticipantSiteEditorProps {
  site: SiteView
  onClose: () => void
  onSave: (site: SiteView) => Promise<void>
}

const validationSchema = Yup.object({
  name: Yup.string().required('Site name is required'),
  num_of_gpus: Yup.number().typeError('Enter a number').integer('Enter a whole number').min(0, 'Must be nonnegative').required('Number of GPUs is required'),
  mem_per_gpu_in_GiB: Yup.number().typeError('Enter a number').min(0, 'Must be nonnegative').required('Memory per GPU is required'),
})

export default function ParticipantSiteEditor({ site, onClose, onSave }: ParticipantSiteEditorProps) {
  return (
    <Formik
      key={site.uid}
      initialValues={site}
      validationSchema={validationSchema}
      onSubmit={async (values, helpers) => {
        helpers.setStatus(undefined)
        try {
          await onSave(values)
          onClose()
        } catch (error) {
          helpers.setStatus(getErrorStatus(error) === 401 ? 'Your session has expired. Please log in again.' : getErrorMessage(error, 'Unable to save the client site. Please check the entered values.'))
        } finally {
          helpers.setSubmitting(false)
        }
      }}
    >
      {formik => (
        <Form>
          <Stack spacing={2} sx={{ pt: 1 }}>
            {formik.status && <Alert severity="error">{formik.status}</Alert>}
            <TextField
              label="Site Name"
              name="name"
              value={formik.values.name}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              disabled={formik.isSubmitting || Number(site.approval_state) >= 100}
              error={Boolean(formik.touched.name && formik.errors.name)}
              helperText={formik.touched.name && formik.errors.name}
              fullWidth
            />
            <TextField
              label="Number of GPUs"
              name="num_of_gpus"
              type="number"
              value={formik.values.num_of_gpus ?? ''}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              disabled={formik.isSubmitting}
              error={Boolean(formik.touched.num_of_gpus && formik.errors.num_of_gpus)}
              helperText={formik.touched.num_of_gpus && formik.errors.num_of_gpus}
              fullWidth
            />
            <TextField
              label="Memory per GPU (GiB)"
              name="mem_per_gpu_in_GiB"
              type="number"
              value={formik.values.mem_per_gpu_in_GiB ?? ''}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              disabled={formik.isSubmitting}
              error={Boolean(formik.touched.mem_per_gpu_in_GiB && formik.errors.mem_per_gpu_in_GiB)}
              helperText={formik.touched.mem_per_gpu_in_GiB && formik.errors.mem_per_gpu_in_GiB}
              fullWidth
            />
            <Stack direction="row" spacing={1} sx={{ justifyContent: 'flex-end' }}>
              <Button onClick={onClose} disabled={formik.isSubmitting}>Cancel</Button>
              <Button type="submit" variant="contained" disabled={formik.isSubmitting}>Save</Button>
            </Stack>
          </Stack>
        </Form>
      )}
    </Formik>
  )
}
