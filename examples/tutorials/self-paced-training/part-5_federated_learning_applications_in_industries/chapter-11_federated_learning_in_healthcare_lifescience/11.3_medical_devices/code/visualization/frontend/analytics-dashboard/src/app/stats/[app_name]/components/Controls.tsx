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

import { Select, SelectItem, SelectOption } from "@kui-react/select";
import { DatepickerRange, DatepickerTrigger } from "@kui-react/datepicker";
import { TextInput } from "@kui-react/text-input";
import { Checkbox } from '@kui-react/checkbox';
import { Label } from '@kui-react/label';
import { Flex } from '@kui-react/flex';
import { DateRange } from "react-day-picker";

type ControlsProps = {
  selectedStatsType: string;
  onStatsTypeChange: (value: string) => void;
  statsList: string[] | null;
  onDateChange: (value: string) => void;
  dateRange: DateRange | undefined;
  onDayClick: (range: DateRange) => void;
  dateString: string;
  expand: boolean;
  onExpand: () => void;
  formatDateString: (str: string) => string;
}

export const Controls = ({
  selectedStatsType,
  onStatsTypeChange,
  statsList,
  onDateChange,
  dateRange,
  onDayClick,
  dateString,
  expand,
  onExpand,
  formatDateString
}: ControlsProps) => (
  <Flex css={{margin: 50}}>
    <Select
      defaultValue={selectedStatsType}
      onChange={onStatsTypeChange}
      options={[
        { label: 'Global Statistics', value: 'global' },
        { label: 'Hierarchical Statistics', value: 'local' }
      ]}
      renderItem={(item: SelectOption, index: number) => (
        <SelectItem key={index} value={item.value}>{item.label}</SelectItem>
      )}
      label="Statistics Type"
      css={{width: 200}}
    />
    {statsList && (
      <Select
        defaultValue={"Latest Statistics"}
        onChange={onDateChange}
        options={statsList.sort().map(str => ({
          'label': str === "Latest Statistics" ? str : formatDateString(str),
          'value': str
        }))}
        renderItem={(item: SelectOption, index: number) => (
          <SelectItem key={index} value={item.value}>{item.label}</SelectItem>
        )}
        label="Select Stats"
        css={{marginLeft: 50, width: 200}}
      />
    )}
    <DatepickerRange
      popoverRootProps={{ defaultOpen: false }}
      selected={dateRange}
      onDayClick={onDayClick}
      numberOfMonths={2}
    >
      <div style={{ maxWidth: '250px', marginTop: 4, marginLeft: 50 }}>
        <TextInput
          label="Select date range"
          value={dateString}
          readOnly
          slotRight={<DatepickerTrigger />}
        />
      </div>
    </DatepickerRange>
    <Flex align="center" gap="sm" css={{marginLeft: 50}}>
      <Label htmlFor="expand">Visualize</Label>
      <Checkbox
        checked={expand}
        id="expand"
        name="expand"
        value="expand"
        onChange={onExpand}
      />
    </Flex>
  </Flex>
);
