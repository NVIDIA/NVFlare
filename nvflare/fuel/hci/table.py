# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional


def repeat_to_length(string_to_expand, length):
    """Repeats string_to_expand to fill up a string of the provided length.

    Args:
        string_to_expand: string to repeat
        length: length of string to return

    Returns: generated string of provided length

    """
    return (string_to_expand * (int(length / len(string_to_expand)) + 1))[:length]


class Table(object):
    def __init__(self, headers: Optional[List[str]] = None, meta_rows=None):
        """A structure with header and rows of records.

        Note:
            The header will be converted to capital letters.

        Args:
            headers: headers of the table
        """
        self.rows = []
        self.meta_rows = meta_rows
        if headers and len(headers) > 0:
            new_headers = []
            for h in headers:
                new_headers.append(h.upper())
            self.rows.append(new_headers)

    def set_rows(self, rows, meta_rows=None):
        """Sets the rows of records."""
        self.rows = rows
        self.meta_rows = meta_rows

    def add_row(self, row: List[str], meta: dict = None):
        """Adds a record."""
        self.rows.append(row)
        if meta:
            self.meta_rows.append(meta)

    def write(self, writer):
        # compute the number of cols
        num_cols = 0
        for row in self.rows:
            if num_cols < len(row):
                num_cols = len(row)

        # compute max col size
        col_len = [0 for _ in range(num_cols)]
        for row in self.rows:
            for i in range(len(row)):
                if col_len[i] < len(row[i]):
                    col_len[i] = len(row[i])

        col_fmt = ["" for _ in range(num_cols)]
        for i in range(num_cols):
            if i == 0:
                extra = ""
            else:
                extra = " "

            col_fmt[i] = extra + "| {:" + "{}".format(col_len[i]) + "}"
            if i == num_cols - 1:
                col_fmt[i] = col_fmt[i] + " |"

        total_col_size = 0
        for i in range(num_cols):
            total_col_size += col_len[i] + 2

        table_width = total_col_size + num_cols + 1
        border_line = repeat_to_length("-", table_width)
        writer.write(border_line + "\n")

        for r in range(len(self.rows)):
            row = self.rows[r]
            line = ""
            for i in range(num_cols):
                if i < len(row):
                    data = row[i]
                else:
                    data = " "

                line += col_fmt[i].format(data)

            writer.write(line + "\n")

            if r == 0:
                writer.write(border_line + "\n")

        writer.write(border_line + "\n")
