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

export type HistogramBucket = [number, number, number];

export interface HoloscanSet<T> {
  [key: string]: T | HistogramBucket[];
}

export interface GlobalMetrics {
  count: {
    holoscan_set: HoloscanSet<number>;
  };
  failure_count: {
    holoscan_set: HoloscanSet<number>;
  };
  sum: {
    holoscan_set: HoloscanSet<number>;
  };
  mean: {
    holoscan_set: HoloscanSet<number>;
  };
  min: {
    holoscan_set: HoloscanSet<number>;
  };
  max: {
    holoscan_set: HoloscanSet<number>;
  };
  histogram: {
    holoscan_set: HoloscanSet<HistogramBucket[]>;
  };
  var: {
    holoscan_set: HoloscanSet<number>;
  };
  stddev: {
    holoscan_set: HoloscanSet<number>;
  };
}

export interface TreeNode {
  Name: string;
  [key: string]: string | TreeNode[] | GlobalMetrics | unknown;
}

export interface DataTree {
  [key: string]: TreeNode[] | GlobalMetrics | Record<string, unknown>;
}

export type DynamicData<T = unknown> = {
  [key: string]: T | DynamicData<T> | Array<T>;
};

export type RootData = DynamicData<GlobalMetrics>;
