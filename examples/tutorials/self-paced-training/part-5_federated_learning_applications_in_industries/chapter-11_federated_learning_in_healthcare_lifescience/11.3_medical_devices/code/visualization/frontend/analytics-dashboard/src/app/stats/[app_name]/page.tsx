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

'use client';
import { useCallback, useState, useEffect } from "react";
import { Text } from "@kui-react/text";
import { ThemeProvider } from "@kui-react/theme";
import { Flex } from '@kui-react/flex';
import { DateRange } from "react-day-picker";
import { format } from 'date-fns';
import { AppNames } from "../../../../src/config";
import { Controls } from './components/Controls';
import { StatisticsTable } from './components/StatisticsTable';
import { Select, SelectItem, SelectOption } from "@kui-react/select";
import { GlobalMetrics, RootData, DataTree, TreeNode } from './types';
import { extractHierarchy, formatDate, formatDateString } from './utils';

type GenericObject = {
  [key: string]: unknown;
};

const StatsPage = ({ params }: { params: { app_name: string } }) => {
  const [renderCurrentLevcelStatistics, setRenderCurrentLevcelStatistics] = useState<boolean>(false);
  const [globalData, setGlobalData] = useState<GlobalMetrics | null>(null);
  const [localData, setLocalData] = useState<RootData | null>(null)
  const [hierarchy, setHierarchy] = useState<GenericObject | null>(null)
  const [statsList, setStatsList] = useState<string[] | null>(null);
  const [columns, setColumns] = useState<string[] | null>(null);
  const [features, setFeatures] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<DateRange | undefined>({
    from: undefined, // can be undefined if not preferred to start from today
    to: undefined,
  })
  const [expand, setExpand] = useState<boolean>(true);
  const [selectedDate, setSelectedDate] = useState<string>("Latest Statistics")
  const [selectedStatsType, setSelectedStatsType] = useState<string>("global");
  const [selectedValues, setSelectedValues] = useState<string[]>([]);

  const isValueAtFinalLevel = (data: unknown, targetValue: unknown): boolean => {
    /**
     * Determines if the targetValue exists at the final level of the hierarchy.
     *
     * @param data - The hierarchical data to search through.
     * @param targetValue - The value to find in the hierarchy.
     * @returns True if the value is at the final level, False otherwise.
     */
    if (typeof data === "object" && data !== null) {
      if (Array.isArray(data)) {
        // Search each item in the array
        for (const item of data) {
          if (isValueAtFinalLevel(item, targetValue)) {
            return true;
          }
        }
      } else {
        // Search each key-value pair in the object
        for (const [, value] of Object.entries(data)) {
          if (value === targetValue) {
            // Check if this key has no nested structures
            return Object.values(data).every(v => typeof v !== "object" || v === null);
          } else if (typeof value === "object" && value !== null) {
            // Recursively search nested objects
            if (isValueAtFinalLevel(value, targetValue)) {
              return true;
            }
          }
        }
      }
    }
    return false; // Return false if the value is not found or is not at the final level
  }

  const handleSelectionChange = (level: number, value: string) => {
    // Update the array of selected values
    const newSelectedValues = [...selectedValues];
    newSelectedValues[level] = value;
    // Reset all subsequent selections
    newSelectedValues.length = level + 1;
    setSelectedValues(newSelectedValues);
    if (value === 'Show Global Statistics' || isValueAtFinalLevel(hierarchy, value)) {
      setRenderCurrentLevcelStatistics(true);
    } else {
      setRenderCurrentLevcelStatistics(false);
    }
  };

  const renderSelectBoxes = (
    data: TreeNode[],
    level: number,
    title: string = ''
  ): JSX.Element | null => {
    if (!data || data.length === 0) {
      return null;
    }

    const currentSelectedValue = selectedValues[level] || '';
    let options = data.map(item => item.Name);

    // Determine subData for the next level
    const selectedItem = data.find(item => item.Name === currentSelectedValue);
    let subData: TreeNode[] = [];
    let subTitle: string = ''
    if (selectedItem) {
      const keys = Object.keys(selectedItem).filter(key => key !== 'Name');
      if (keys.length > 0 && Array.isArray(selectedItem[keys[0]])) {
        subTitle = keys[0];
        subData = JSON.parse(JSON.stringify(selectedItem[keys[0]]));
      }
    }
    if (!isValueAtFinalLevel(hierarchy, options[0])) {
      options = [...options, 'Show Global Statistics']
    }
    return (
      <>
        <Select
          defaultValue={currentSelectedValue}
          onChange={(value: string) => handleSelectionChange(level, value)}
          options={JSON.parse(JSON.stringify(options.map((str => ({
            'label': str,
            'value': str
          })))))}
          renderItem={(item: SelectOption, index: number) => (
            <SelectItem
              key={index}
              value={item.value}
            >
              {item.label}
            </SelectItem>
          )}
          label={`Select ${title}`}
          css={{marginLeft: 50, width: 200}}
        />
        {subData.length > 0 && renderSelectBoxes(subData, level + 1, subTitle)}
      </>
    );
  };

  const handleDateChange = (value: string) => {
    setSelectedDate(value); // Access the selected value
    setDateRange(undefined);
    if (value == "Latest Statistics" && selectedDate != "Latest Statistics") {
      fetchStats();
    } else {
      fetchStats(value);
    }
  };

  const handleStatsTypeChange = (value: string) => {
    setSelectedStatsType(value); // Access the selected value
    setDateRange(undefined);
    setSelectedValues([]);
  };

  const handleOnDayClick = (range: DateRange) => {
     if(range.to && range.from) {
       setDateRange(range);
       fetchStats(range)
     }
  }

  const onExpand = () => {
     setExpand(!expand) ;
  }

  let dateString = ''
  if (dateRange?.from && dateRange.to) {
    dateString = `${format(new Date(dateRange.from), 'P')}-${format(
      new Date(dateRange.to),
      'P'
    )}`
  }

  // Fetch data from visualization backend API using useEffect
  const fetchStats = useCallback((date: string | DateRange | undefined = undefined) => {
    (async () => {
      try {
        const authorizationHeader = `Bearer ${process.env.NEXT_PUBLIC_AUTHORIZATION_HEADER}`;
        let request_uri = `${process.env.NEXT_PUBLIC_ROOT_URI}/get_stats/${params.app_name}`;
        if(date && typeof(date) === 'string') {
          request_uri = `${request_uri}/?timestamp=${date}`;
        } else if (date && (date as DateRange)?.to && (date as DateRange)?.from) {
          const fr = (date as DateRange)?.from;
          const to = (date as DateRange)?.to;
          if (fr && to) {
            request_uri = `${process.env.NEXT_PUBLIC_ROOT_URI}/get_range_stats/${params.app_name}/${formatDate(fr)}/${formatDate(to)}/`;
          }
        }
        const res = await fetch(request_uri, {
          method: 'GET',
          headers: {
            'Authorization': `${authorizationHeader}`,
            'Content-Type': 'application/json'
          }
        });
        if (!res.ok) {
          throw new Error(`An error occurred while fetching the stats for the given apps: ${res.statusText}`);
        }
        const result = await res.json();
        if (Object.keys(result).length > 0) {
          setGlobalData(result.Global);
          setColumns(['Feature / Metric', ...Object.keys(result.Global)]);
          setFeatures(Object.keys(result.Global[Object.keys(result.Global)[0]]?.holoscan_set));

          // Get the JSON data without high level global stats and calculate hierarchy
          const entries = Object.entries(result);
          const [, ...restEntries] = entries;
          const hierarchyData = extractHierarchy(JSON.parse(JSON.stringify(Object.fromEntries(restEntries))));
          const root: RootData = JSON.parse(JSON.stringify(Object.fromEntries(restEntries)));
          setLocalData(root);
          setHierarchy(hierarchyData);
        } else {
          setDateRange(undefined);
        }
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError((err as Error).message);
        }
      }
    })();
  }, [params.app_name]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

    // Fetch data from visualization backend API using useEffect
    const fetchStatsList = useCallback(() => {
      (async () => {
        try {
          const authorizationHeader = `Bearer ${process.env.NEXT_PUBLIC_AUTHORIZATION_HEADER}`;
          const res = await fetch(`${process.env.NEXT_PUBLIC_ROOT_URI}/get_stats_list/${params.app_name}`, {
              method: 'GET',
              headers: {
                  'Authorization': `${authorizationHeader}`,
                  'Content-Type': 'application/json'
              }
          });
          if (!res.ok) {
            throw new Error(`An error occurred while fetching the stats for the given apps: ${res.statusText}`);
          }
          const result = await res.json();
          if (selectedDate == "Latest Statistics") {
            setStatsList([selectedDate, ...result]);
          } else {
            setStatsList([...result, "Latest Statistics"]);
          }
        } catch (err: unknown) {
          if (err instanceof Error) {
            setError((err as Error).message);  // Store error in state
          }
        }
      })();
    }, [params.app_name, selectedDate]);

  useEffect(() => {
      fetchStatsList();
    }, [fetchStatsList]);

  // Display loading, error or data
  return (
    <ThemeProvider theme="light" withFonts withReset>
      <div>
        {error && <p>Error: {error}</p>}
        {!globalData && !error && <p>Loading...</p>}
        {globalData && (
          <div style={{width: '100%', textAlign: 'center'}}>
            <Text variant="h1">{AppNames.get(params.app_name)}</Text>
            <Controls
              selectedStatsType={selectedStatsType}
              onStatsTypeChange={handleStatsTypeChange}
              statsList={statsList}
              onDateChange={handleDateChange}
              dateRange={dateRange}
              onDayClick={handleOnDayClick}
              dateString={dateString}
              expand={expand}
              onExpand={onExpand}
              formatDateString={formatDateString}
            />
            {selectedStatsType === "global" && (
              <StatisticsTable
                columns={columns}
                features={features}
                data={globalData}
                expand={expand}
                variant="global"
              />
            )}
            {/* Render local statistics */}
            {selectedStatsType === "local" && hierarchy &&
              <>
                <Flex css={{marginTop: -50}}>
                  {renderSelectBoxes(JSON.parse(JSON.stringify(hierarchy[Object.keys(hierarchy)[1]])), 0, Object.keys(hierarchy)[1])}
                </Flex>
                <Flex>
                  {(() => {
                    let index = 0;
                    let dataTree: DataTree | null = localData as DataTree;
                    const len = selectedValues.length;

                    while (index <= len - 1 && selectedValues[index] !== 'Show Global Statistics' && dataTree) {
                      const firstKey = Object.keys(dataTree)[0];
                      const children = dataTree[firstKey] as TreeNode[];
                      const found = children.find(child => child.Name === selectedValues[index]);

                      if (found) {
                        const entries = Object.entries(found);
                        const [, ...restEntries] = entries;
                        dataTree = Object.fromEntries(restEntries) as DataTree;
                      } else {
                        dataTree = null;
                      }
                      index++;
                    }

                    let tempData: GlobalMetrics = globalData as GlobalMetrics;

                    if (dataTree?.Global) {
                      tempData = dataTree.Global as GlobalMetrics;
                    } else if (dataTree?.Local) {
                      tempData = dataTree.Local as GlobalMetrics;
                    }

                    if (renderCurrentLevcelStatistics) {
                      return <StatisticsTable
                        columns={columns}
                        features={features}
                        data={tempData}
                        expand={expand}
                        variant="local"
                      />;
                    }
                    return <></>;
                  })()}
                </Flex>
              </>
            }
          </div>
        )}
      </div>
    </ThemeProvider>
  );
};

export default StatsPage;
