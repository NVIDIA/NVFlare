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

"""Generate the shifted FedRevive paper plan before launching SimEnv."""

from functools import partial

from schedule_standard import main as standard_main
from schedule_standard import make_paper_plan as make_standard_plan
from schedule_standard import sample_paper_delay

# The shifted manifest supplies the per-client means. The upload distribution
# has a smaller default half-width for profiles that omit it.
sample_shifted_delay = partial(sample_paper_delay, default_upload_half_width=0.01)
make_paper_plan = partial(make_standard_plan, paper_delay=sample_shifted_delay)
main = partial(standard_main, expected_delay_schedule="shifted", planner=make_paper_plan)


if __name__ == "__main__":
    main()
