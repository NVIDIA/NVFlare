/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { RootData } from './types';

export const formatDate = (date: Date): string => {
  const yyyy = date.getFullYear();
  const MM = String(date.getMonth() + 1).padStart(2, "0");
  const DD = String(date.getDate()).padStart(2, "0");
  const HH = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  const SS = String(date.getSeconds()).padStart(2, "0");

  return `${yyyy}${MM}${DD}_${HH}${mm}${SS}`;
};

export const formatDateString = (dateString: string): string => {
  const regex = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/;
  const match = dateString.match(regex);

  if (!match) {
    throw new Error("Input date format should be YYYYMMDD_HHMMSS");
  }

  const [, year, month, day, hour, minute, second] = match;
  return `${year}/${month}/${day} ${hour}:${minute}:${second}`;
};

export function extractHierarchy(data: RootData): RootData {
  if (Array.isArray(data)) {
    return JSON.parse(JSON.stringify(data.map(extractHierarchy)));
  } else if (typeof data === 'object' && data !== null) {
    const result: RootData = { Name: data.Name };
    for (const key in data) {
      if (Array.isArray(data[key])) {
        result[key] = extractHierarchy(JSON.parse(JSON.stringify(data[key])));
      }
    }
    return result;
  }
  return data;
}
