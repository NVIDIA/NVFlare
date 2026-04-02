# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Static lookup values for payment dataset fields.

These constants define the fixed vocabularies used when generating categorical
attributes such as payment status and account type.
"""

PAYMENT_CREDITOR_PREFIX = "CREDITOR"
"""Column-name prefix for creditor-side attributes."""

PAYMENT_DEBTOR_PREFIX = "DEBITOR"
"""Column-name prefix for debitor-side attributes."""

PAYMENT_STATUS = ("PENDING", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED")
"""Valid payment status values."""

ACCOUNT_TYPES = ("SAVINGS", "CHECKING", "BUSINESS")
"""Valid account type values."""
