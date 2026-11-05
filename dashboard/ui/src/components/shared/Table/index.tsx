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

import { Alert, Box } from '@mui/material'
import { DataGrid, GridCallbackDetails, GridColDef, GridRowIdGetter, GridRowSelectionModel, GridValidRowModel } from '@mui/x-data-grid'

export interface TableProps<T extends GridValidRowModel> {
  columns: GridColDef<T>[]
  data: T[]
  onRowClick?: (row: T) => void
  getRowId?: GridRowIdGetter<T>
  checkboxSelection?: boolean
  onSelectionModelChange?: (selection: GridRowSelectionModel, details: GridCallbackDetails) => void
  selectionModel?: GridRowSelectionModel
  disableColumnMenu?: boolean
  disableColumnSelector?: boolean
  disableDensitySelector?: boolean
  disableColumnFilter?: boolean
  disableRowSelectionOnClick?: boolean
  loading?: boolean
  error?: Error | null
  pageSize?: number
  pageSizeOptions?: number[]
}

function Table<T extends GridValidRowModel>({
  columns, data, onRowClick, getRowId, checkboxSelection = false,
  onSelectionModelChange, selectionModel, disableColumnMenu = true,
  disableColumnSelector = true, disableDensitySelector = true,
  disableColumnFilter = false, disableRowSelectionOnClick = true,
  loading = false, error, pageSize = 10, pageSizeOptions = [10, 25, 50],
}: TableProps<T>) {
  return (
    <Box sx={{ width: '100%', minWidth: 0 }}>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error.message}</Alert>}
      <Box sx={{ height: 460, width: '100%' }}>
        <DataGrid<T>
          rows={data}
          columns={columns}
          getRowId={getRowId}
          onRowClick={params => onRowClick?.(params.row)}
          checkboxSelection={checkboxSelection}
          onRowSelectionModelChange={onSelectionModelChange}
          rowSelectionModel={selectionModel}
          disableRowSelectionExcludeModel
          disableColumnMenu={disableColumnMenu}
          disableColumnSelector={disableColumnSelector}
          disableDensitySelector={disableDensitySelector}
          disableColumnFilter={disableColumnFilter}
          disableRowSelectionOnClick={disableRowSelectionOnClick}
          loading={loading}
          initialState={{ pagination: { paginationModel: { pageSize, page: 0 } } }}
          pageSizeOptions={pageSizeOptions}
          sx={{ border: 0, '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 600 } }}
        />
      </Box>
    </Box>
  )
}

export default Table
