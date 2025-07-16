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

import React from 'react';
import { Table, TableRow, TableHeader, TableDataCell } from '@kui-react/table';
import { GlobalMetrics } from '../types';
import { HistogramRow } from './HistogramRow';

interface StatisticsTableProps {
  columns: string[] | null;
  features: string[] | null;
  data: GlobalMetrics;
  expand: boolean;
  variant?: 'global' | 'local';
}

export const StatisticsTable = ({
  columns,
  features,
  data,
  expand,
  variant = 'global'
}: StatisticsTableProps) => {
  const styles = {
    container: {
      marginTop: variant === 'global' ? -50 : 0,
      marginLeft: 50
    },
    table: {
      width: variant === 'global' ? '80%' : '100%'
    }
  };

  return (
    <div style={styles.container}>
      <Table css={styles.table}>
        <thead>
          <TableRow selected css={{backgroundColor: '#76b900' }}>
            {columns?.map(header => (
              <TableHeader key={header}>{header}</TableHeader>
            ))}
          </TableRow>
        </thead>
        <tbody>
          {features?.map(featureRow => (
            <React.Fragment key={featureRow}>
              <TableRow selected css={{width: '100%'}}>
                <TableHeader>{featureRow}</TableHeader>
              </TableRow>
              <TableRow css={{backgroundColor: '#F6FFEB'}}>
                {columns?.map((col: string) => (
                  <TableDataCell key={col}>
                    {JSON.stringify(data[col as keyof GlobalMetrics]?.holoscan_set[featureRow])}
                  </TableDataCell>
                ))}
              </TableRow>
              {expand && <HistogramRow data={data} featureRow={featureRow} />}
            </React.Fragment>
          ))}
        </tbody>
      </Table>
    </div>
  );
};
