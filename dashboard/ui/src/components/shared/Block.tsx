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

import { ReactNode } from 'react'
import { Box, CircularProgress, Paper, Typography } from '@mui/material'

export default function Block({ title, children, actions, className, loading }: { title?: string; children: ReactNode; actions?: ReactNode; className?: string; loading?: boolean }) {
  return <Paper className={className} sx={{ p: 3, mb: 3, position: 'relative' }}>
    {(title || actions) && <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}><Typography variant="h6" component="h2">{title}</Typography>{actions}</Box>}
    {children}
    {loading && <Box sx={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', bgcolor: 'action.disabledBackground' }}><CircularProgress aria-label="Loading" /></Box>}
  </Paper>
}
