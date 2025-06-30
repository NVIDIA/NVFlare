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

import { TableRow, TableDataCell } from '@kui-react/table';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { GlobalMetrics } from '../types';

interface HistogramRowProps {
  data: GlobalMetrics;
  featureRow: string;
}

export const HistogramRow = ({ data, featureRow }: HistogramRowProps) => (
  <TableRow css={{alignText: 'center'}}>
    <TableDataCell></TableDataCell>
    <TableDataCell></TableDataCell>
    <TableDataCell></TableDataCell>
    <TableDataCell></TableDataCell>
    <TableDataCell></TableDataCell>
    <TableDataCell></TableDataCell>
    <TableDataCell></TableDataCell>
    {(() => {
      const hist = data["histogram"]?.holoscan_set[featureRow];
      if (!hist) return null;
      const histogramData = hist.map(([start, end, count]) => ({
        range: `${start} - ${end}`,
        count,
      }));
      return (
        <ResponsiveContainer width="80%" height={200}>
          <BarChart margin={{ top: 20, right: 30, left: 30, bottom: 20}} data={histogramData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" label={{ value: "Range", position: "insideBottom", offset: -5 }} />
            <YAxis label={{ value: "Count", angle: -90, position: "insideLeft", offset: -5 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#76b900" barSize={30}/>
          </BarChart>
        </ResponsiveContainer>
      );
    })()}
  </TableRow>
);
