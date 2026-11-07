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

export type ErrorData = {
  message?: string
  errors?: Record<string, string>
}

type ResponseError = {
  response?: { status?: number; data?: ErrorData | Blob }
  message?: string
}

function errorDetails(error: unknown): ResponseError {
  return error !== null && typeof error === 'object' ? error as ResponseError : {}
}

export function getErrorStatus(error: unknown): number | undefined {
  return errorDetails(error).response?.status
}

export function getValidationErrors(error: unknown): Record<string, string> | undefined {
  const response = errorDetails(error).response
  if (response?.status === 400 && response.data && !(response.data instanceof Blob)) {
    return response.data.errors
  }
}

export function getErrorMessage(error: unknown, fallback: string): string {
  const details = errorDetails(error)
  const data = details.response?.data
  if (data && !(data instanceof Blob) && data.message) return data.message
  return details.message || fallback
}

export async function getDownloadErrorMessage(error: unknown): Promise<string> {
  const fallback = 'Unable to download the startup kit. Please try again or contact the project administrator.'
  let data = errorDetails(error).response?.data
  if (data instanceof Blob) {
    try {
      data = JSON.parse(await data.text()) as ErrorData
    } catch {
      return fallback
    }
  }
  return data?.message || errorDetails(error).message || fallback
}
