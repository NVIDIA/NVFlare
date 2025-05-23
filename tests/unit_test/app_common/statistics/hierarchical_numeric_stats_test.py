# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.workflows.statistics_controller import StatisticsController

# Unit test upto four hierarchical levels
HIERARCHY_CONFIGS = [
    {"Sites": ["Site-1", "Site-2", "Site-3", "Site-4"]},
    {
        "Manufacturers": [
            {"Name": "Manufacturer-1", "Sites": ["Site-1", "Site-2"]},
            {"Name": "Manufacturer-2", "Sites": ["Site-3", "Site-4"]},
        ]
    },
    {
        "Manufacturers": [
            {
                "Name": "Manufacturer-1",
                "Orgs": [{"Name": "Org-1", "Sites": ["Site-1"]}, {"Name": "Org-2", "Sites": ["Site-2"]}],
            },
            {"Name": "Manufacturer-2", "Orgs": [{"Name": "Org-3", "Sites": ["Site-3", "Site-4"]}]},
        ]
    },
    {
        "Manufacturers": [
            {
                "Name": "Manufacturer-1",
                "Orgs": [
                    {"Name": "Org-1", "Locations": [{"Name": "Location-1", "Sites": ["Site-1"]}]},
                    {"Name": "Org-2", "Locations": [{"Name": "Location-1", "Sites": ["Site-2"]}]},
                ],
            },
            {
                "Name": "Manufacturer-2",
                "Orgs": [
                    {
                        "Name": "Org-3",
                        "Locations": [
                            {"Name": "Location-1", "Sites": ["Site-3"]},
                            {"Name": "Location-2", "Sites": ["Site-4"]},
                        ],
                    }
                ],
            },
        ]
    },
]

hist_bins = []
hist_bins.append(Bin(0.0, 0.5, 50))
hist_bins.append(Bin(0.5, 1.0, 50))
g_hist = Histogram(HistogramType.STANDARD, hist_bins)

CLIENT_STATS = {
    "count": {
        "Site-1": {"data_set1": {"Feature1": 100}},
        "Site-2": {"data_set1": {"Feature1": 200}},
        "Site-3": {"data_set1": {"Feature1": 300}},
        "Site-4": {"data_set1": {"Feature1": 400}},
    },
    "sum": {
        "Site-1": {"data_set1": {"Feature1": 1000}},
        "Site-2": {"data_set1": {"Feature1": 2000}},
        "Site-3": {"data_set1": {"Feature1": 3000}},
        "Site-4": {"data_set1": {"Feature1": 4000}},
    },
    "max": {
        "Site-1": {"data_set1": {"Feature1": 20}},
        "Site-2": {"data_set1": {"Feature1": 30}},
        "Site-3": {"data_set1": {"Feature1": 40}},
        "Site-4": {"data_set1": {"Feature1": 50}},
    },
    "min": {
        "Site-1": {"data_set1": {"Feature1": 0}},
        "Site-2": {"data_set1": {"Feature1": 1}},
        "Site-3": {"data_set1": {"Feature1": 2}},
        "Site-4": {"data_set1": {"Feature1": 3}},
    },
    "mean": {
        "Site-1": {"data_set1": {"Feature1": 10}},
        "Site-2": {"data_set1": {"Feature1": 10}},
        "Site-3": {"data_set1": {"Feature1": 10}},
        "Site-4": {"data_set1": {"Feature1": 10}},
    },
    "var": {
        "Site-1": {"data_set1": {"Feature1": 0.1}},
        "Site-2": {"data_set1": {"Feature1": 0.1}},
        "Site-3": {"data_set1": {"Feature1": 0.1}},
        "Site-4": {"data_set1": {"Feature1": 0.1}},
    },
    "stddev": {
        "Site-1": {"data_set1": {"Feature1": 0.1}},
        "Site-2": {"data_set1": {"Feature1": 0.1}},
        "Site-3": {"data_set1": {"Feature1": 0.1}},
        "Site-4": {"data_set1": {"Feature1": 0.1}},
    },
    "histogram": {
        "Site-1": {"data_set1": {"Feature1": g_hist}},
        "Site-2": {"data_set1": {"Feature1": g_hist}},
        "Site-3": {"data_set1": {"Feature1": g_hist}},
        "Site-4": {"data_set1": {"Feature1": g_hist}},
    },
}

global_stats_0 = {
    "Global": {
        "count": {"data_set1": {"Feature1": 1000}},
        "sum": {"data_set1": {"Feature1": 10000}},
        "mean": {"data_set1": {"Feature1": 10.0}},
        "min": {"data_set1": {"Feature1": 1}},
        "max": {"data_set1": {"Feature1": 50}},
        "histogram": {
            "data_set1": {
                "Feature1": [
                    Bin(low_value=0.0, high_value=0.5, sample_count=200),
                    Bin(low_value=0.5, high_value=1.0, sample_count=200),
                ]
            }
        },
        "var": {"data_set1": {"Feature1": 0.4}},
        "stddev": {"data_set1": {"Feature1": 0.63}},
    },
    "Sites": [
        {
            "Name": "Site-1",
            "Local": {
                "count": {"data_set1": {"Feature1": 100}},
                "sum": {"data_set1": {"Feature1": 1000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 0}},
                "max": {"data_set1": {"Feature1": 20}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.1}},
                "stddev": {"data_set1": {"Feature1": 0.32}},
            },
        },
        {
            "Name": "Site-2",
            "Local": {
                "count": {"data_set1": {"Feature1": 200}},
                "sum": {"data_set1": {"Feature1": 2000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 1}},
                "max": {"data_set1": {"Feature1": 30}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.1}},
                "stddev": {"data_set1": {"Feature1": 0.32}},
            },
        },
        {
            "Name": "Site-3",
            "Local": {
                "count": {"data_set1": {"Feature1": 300}},
                "sum": {"data_set1": {"Feature1": 3000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 2}},
                "max": {"data_set1": {"Feature1": 40}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.1}},
                "stddev": {"data_set1": {"Feature1": 0.32}},
            },
        },
        {
            "Name": "Site-4",
            "Local": {
                "count": {"data_set1": {"Feature1": 400}},
                "sum": {"data_set1": {"Feature1": 4000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 3}},
                "max": {"data_set1": {"Feature1": 50}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.1}},
                "stddev": {"data_set1": {"Feature1": 0.32}},
            },
        },
    ],
}
global_stats_1 = {
    "Global": {
        "count": {"data_set1": {"Feature1": 1000}},
        "sum": {"data_set1": {"Feature1": 10000}},
        "mean": {"data_set1": {"Feature1": 10.0}},
        "min": {"data_set1": {"Feature1": 1}},
        "max": {"data_set1": {"Feature1": 50}},
        "histogram": {
            "data_set1": {
                "Feature1": [
                    Bin(low_value=0.0, high_value=0.5, sample_count=200),
                    Bin(low_value=0.5, high_value=1.0, sample_count=200),
                ]
            }
        },
        "var": {"data_set1": {"Feature1": 0.4}},
        "stddev": {"data_set1": {"Feature1": 0.63}},
    },
    "Manufacturers": [
        {
            "Name": "Manufacturer-1",
            "Sites": [
                {
                    "Name": "Site-1",
                    "Local": {
                        "count": {"data_set1": {"Feature1": 100}},
                        "sum": {"data_set1": {"Feature1": 1000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 0}},
                        "max": {"data_set1": {"Feature1": 20}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
                {
                    "Name": "Site-2",
                    "Local": {
                        "count": {"data_set1": {"Feature1": 200}},
                        "sum": {"data_set1": {"Feature1": 2000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 1}},
                        "max": {"data_set1": {"Feature1": 30}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
            ],
            "Global": {
                "count": {"data_set1": {"Feature1": 300}},
                "sum": {"data_set1": {"Feature1": 3000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 1}},
                "max": {"data_set1": {"Feature1": 30}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=100),
                            Bin(low_value=0.5, high_value=1.0, sample_count=100),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.2}},
                "stddev": {"data_set1": {"Feature1": 0.45}},
            },
        },
        {
            "Name": "Manufacturer-2",
            "Sites": [
                {
                    "Name": "Site-3",
                    "Local": {
                        "count": {"data_set1": {"Feature1": 300}},
                        "sum": {"data_set1": {"Feature1": 3000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 2}},
                        "max": {"data_set1": {"Feature1": 40}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
                {
                    "Name": "Site-4",
                    "Local": {
                        "count": {"data_set1": {"Feature1": 400}},
                        "sum": {"data_set1": {"Feature1": 4000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 3}},
                        "max": {"data_set1": {"Feature1": 50}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
            ],
            "Global": {
                "count": {"data_set1": {"Feature1": 700}},
                "sum": {"data_set1": {"Feature1": 7000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 2}},
                "max": {"data_set1": {"Feature1": 50}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=100),
                            Bin(low_value=0.5, high_value=1.0, sample_count=100),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.2}},
                "stddev": {"data_set1": {"Feature1": 0.45}},
            },
        },
    ],
}
global_stats_2 = {
    "Global": {
        "count": {"data_set1": {"Feature1": 1000}},
        "sum": {"data_set1": {"Feature1": 10000}},
        "mean": {"data_set1": {"Feature1": 10.0}},
        "min": {"data_set1": {"Feature1": 1}},
        "max": {"data_set1": {"Feature1": 50}},
        "histogram": {
            "data_set1": {
                "Feature1": [
                    Bin(low_value=0.0, high_value=0.5, sample_count=200),
                    Bin(low_value=0.5, high_value=1.0, sample_count=200),
                ]
            }
        },
        "var": {"data_set1": {"Feature1": 0.4}},
        "stddev": {"data_set1": {"Feature1": 0.63}},
    },
    "Manufacturers": [
        {
            "Name": "Manufacturer-1",
            "Orgs": [
                {
                    "Name": "Org-1",
                    "Sites": [
                        {
                            "Name": "Site-1",
                            "Local": {
                                "count": {"data_set1": {"Feature1": 100}},
                                "sum": {"data_set1": {"Feature1": 1000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 0}},
                                "max": {"data_set1": {"Feature1": 20}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        }
                    ],
                    "Global": {
                        "count": {"data_set1": {"Feature1": 100}},
                        "sum": {"data_set1": {"Feature1": 1000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 0}},
                        "max": {"data_set1": {"Feature1": 20}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
                {
                    "Name": "Org-2",
                    "Sites": [
                        {
                            "Name": "Site-2",
                            "Local": {
                                "count": {"data_set1": {"Feature1": 200}},
                                "sum": {"data_set1": {"Feature1": 2000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 1}},
                                "max": {"data_set1": {"Feature1": 30}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        }
                    ],
                    "Global": {
                        "count": {"data_set1": {"Feature1": 200}},
                        "sum": {"data_set1": {"Feature1": 2000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 1}},
                        "max": {"data_set1": {"Feature1": 30}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
            ],
            "Global": {
                "count": {"data_set1": {"Feature1": 300}},
                "sum": {"data_set1": {"Feature1": 3000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 1}},
                "max": {"data_set1": {"Feature1": 30}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=100),
                            Bin(low_value=0.5, high_value=1.0, sample_count=100),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.2}},
                "stddev": {"data_set1": {"Feature1": 0.45}},
            },
        },
        {
            "Name": "Manufacturer-2",
            "Orgs": [
                {
                    "Name": "Org-3",
                    "Sites": [
                        {
                            "Name": "Site-3",
                            "Local": {
                                "count": {"data_set1": {"Feature1": 300}},
                                "sum": {"data_set1": {"Feature1": 3000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 2}},
                                "max": {"data_set1": {"Feature1": 40}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        },
                        {
                            "Name": "Site-4",
                            "Local": {
                                "count": {"data_set1": {"Feature1": 400}},
                                "sum": {"data_set1": {"Feature1": 4000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 3}},
                                "max": {"data_set1": {"Feature1": 50}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        },
                    ],
                    "Global": {
                        "count": {"data_set1": {"Feature1": 700}},
                        "sum": {"data_set1": {"Feature1": 7000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 2}},
                        "max": {"data_set1": {"Feature1": 50}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=100),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=100),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.2}},
                        "stddev": {"data_set1": {"Feature1": 0.45}},
                    },
                }
            ],
            "Global": {
                "count": {"data_set1": {"Feature1": 700}},
                "sum": {"data_set1": {"Feature1": 7000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 2}},
                "max": {"data_set1": {"Feature1": 50}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=100),
                            Bin(low_value=0.5, high_value=1.0, sample_count=100),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.2}},
                "stddev": {"data_set1": {"Feature1": 0.45}},
            },
        },
    ],
}
global_stats_3 = {
    "Global": {
        "count": {"data_set1": {"Feature1": 1000}},
        "sum": {"data_set1": {"Feature1": 10000}},
        "mean": {"data_set1": {"Feature1": 10.0}},
        "min": {"data_set1": {"Feature1": 1}},
        "max": {"data_set1": {"Feature1": 50}},
        "histogram": {
            "data_set1": {
                "Feature1": [
                    Bin(low_value=0.0, high_value=0.5, sample_count=200),
                    Bin(low_value=0.5, high_value=1.0, sample_count=200),
                ]
            }
        },
        "var": {"data_set1": {"Feature1": 0.4}},
        "stddev": {"data_set1": {"Feature1": 0.63}},
    },
    "Manufacturers": [
        {
            "Name": "Manufacturer-1",
            "Orgs": [
                {
                    "Name": "Org-1",
                    "Locations": [
                        {
                            "Name": "Location-1",
                            "Sites": [
                                {
                                    "Name": "Site-1",
                                    "Local": {
                                        "count": {"data_set1": {"Feature1": 100}},
                                        "sum": {"data_set1": {"Feature1": 1000}},
                                        "mean": {"data_set1": {"Feature1": 10.0}},
                                        "min": {"data_set1": {"Feature1": 0}},
                                        "max": {"data_set1": {"Feature1": 20}},
                                        "histogram": {
                                            "data_set1": {
                                                "Feature1": [
                                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                                ]
                                            }
                                        },
                                        "var": {"data_set1": {"Feature1": 0.1}},
                                        "stddev": {"data_set1": {"Feature1": 0.32}},
                                    },
                                }
                            ],
                            "Global": {
                                "count": {"data_set1": {"Feature1": 100}},
                                "sum": {"data_set1": {"Feature1": 1000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 0}},
                                "max": {"data_set1": {"Feature1": 20}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        }
                    ],
                    "Global": {
                        "count": {"data_set1": {"Feature1": 100}},
                        "sum": {"data_set1": {"Feature1": 1000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 0}},
                        "max": {"data_set1": {"Feature1": 20}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
                {
                    "Name": "Org-2",
                    "Locations": [
                        {
                            "Name": "Location-1",
                            "Sites": [
                                {
                                    "Name": "Site-2",
                                    "Local": {
                                        "count": {"data_set1": {"Feature1": 200}},
                                        "sum": {"data_set1": {"Feature1": 2000}},
                                        "mean": {"data_set1": {"Feature1": 10.0}},
                                        "min": {"data_set1": {"Feature1": 1}},
                                        "max": {"data_set1": {"Feature1": 30}},
                                        "histogram": {
                                            "data_set1": {
                                                "Feature1": [
                                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                                ]
                                            }
                                        },
                                        "var": {"data_set1": {"Feature1": 0.1}},
                                        "stddev": {"data_set1": {"Feature1": 0.32}},
                                    },
                                }
                            ],
                            "Global": {
                                "count": {"data_set1": {"Feature1": 200}},
                                "sum": {"data_set1": {"Feature1": 2000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 1}},
                                "max": {"data_set1": {"Feature1": 30}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        }
                    ],
                    "Global": {
                        "count": {"data_set1": {"Feature1": 200}},
                        "sum": {"data_set1": {"Feature1": 2000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 1}},
                        "max": {"data_set1": {"Feature1": 30}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.1}},
                        "stddev": {"data_set1": {"Feature1": 0.32}},
                    },
                },
            ],
            "Global": {
                "count": {"data_set1": {"Feature1": 300}},
                "sum": {"data_set1": {"Feature1": 3000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 1}},
                "max": {"data_set1": {"Feature1": 30}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=100),
                            Bin(low_value=0.5, high_value=1.0, sample_count=100),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.2}},
                "stddev": {"data_set1": {"Feature1": 0.45}},
            },
        },
        {
            "Name": "Manufacturer-2",
            "Orgs": [
                {
                    "Name": "Org-3",
                    "Locations": [
                        {
                            "Name": "Location-1",
                            "Sites": [
                                {
                                    "Name": "Site-3",
                                    "Local": {
                                        "count": {"data_set1": {"Feature1": 300}},
                                        "sum": {"data_set1": {"Feature1": 3000}},
                                        "mean": {"data_set1": {"Feature1": 10.0}},
                                        "min": {"data_set1": {"Feature1": 2}},
                                        "max": {"data_set1": {"Feature1": 40}},
                                        "histogram": {
                                            "data_set1": {
                                                "Feature1": [
                                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                                ]
                                            }
                                        },
                                        "var": {"data_set1": {"Feature1": 0.1}},
                                        "stddev": {"data_set1": {"Feature1": 0.32}},
                                    },
                                }
                            ],
                            "Global": {
                                "count": {"data_set1": {"Feature1": 300}},
                                "sum": {"data_set1": {"Feature1": 3000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 2}},
                                "max": {"data_set1": {"Feature1": 40}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        },
                        {
                            "Name": "Location-2",
                            "Sites": [
                                {
                                    "Name": "Site-4",
                                    "Local": {
                                        "count": {"data_set1": {"Feature1": 400}},
                                        "sum": {"data_set1": {"Feature1": 4000}},
                                        "mean": {"data_set1": {"Feature1": 10.0}},
                                        "min": {"data_set1": {"Feature1": 3}},
                                        "max": {"data_set1": {"Feature1": 50}},
                                        "histogram": {
                                            "data_set1": {
                                                "Feature1": [
                                                    Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                                    Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                                ]
                                            }
                                        },
                                        "var": {"data_set1": {"Feature1": 0.1}},
                                        "stddev": {"data_set1": {"Feature1": 0.32}},
                                    },
                                }
                            ],
                            "Global": {
                                "count": {"data_set1": {"Feature1": 400}},
                                "sum": {"data_set1": {"Feature1": 4000}},
                                "mean": {"data_set1": {"Feature1": 10.0}},
                                "min": {"data_set1": {"Feature1": 3}},
                                "max": {"data_set1": {"Feature1": 50}},
                                "histogram": {
                                    "data_set1": {
                                        "Feature1": [
                                            Bin(low_value=0.0, high_value=0.5, sample_count=50),
                                            Bin(low_value=0.5, high_value=1.0, sample_count=50),
                                        ]
                                    }
                                },
                                "var": {"data_set1": {"Feature1": 0.1}},
                                "stddev": {"data_set1": {"Feature1": 0.32}},
                            },
                        },
                    ],
                    "Global": {
                        "count": {"data_set1": {"Feature1": 700}},
                        "sum": {"data_set1": {"Feature1": 7000}},
                        "mean": {"data_set1": {"Feature1": 10.0}},
                        "min": {"data_set1": {"Feature1": 2}},
                        "max": {"data_set1": {"Feature1": 50}},
                        "histogram": {
                            "data_set1": {
                                "Feature1": [
                                    Bin(low_value=0.0, high_value=0.5, sample_count=100),
                                    Bin(low_value=0.5, high_value=1.0, sample_count=100),
                                ]
                            }
                        },
                        "var": {"data_set1": {"Feature1": 0.2}},
                        "stddev": {"data_set1": {"Feature1": 0.45}},
                    },
                }
            ],
            "Global": {
                "count": {"data_set1": {"Feature1": 700}},
                "sum": {"data_set1": {"Feature1": 7000}},
                "mean": {"data_set1": {"Feature1": 10.0}},
                "min": {"data_set1": {"Feature1": 2}},
                "max": {"data_set1": {"Feature1": 50}},
                "histogram": {
                    "data_set1": {
                        "Feature1": [
                            Bin(low_value=0.0, high_value=0.5, sample_count=100),
                            Bin(low_value=0.5, high_value=1.0, sample_count=100),
                        ]
                    }
                },
                "var": {"data_set1": {"Feature1": 0.2}},
                "stddev": {"data_set1": {"Feature1": 0.45}},
            },
        },
    ],
}

PARAMS = [
    (CLIENT_STATS, global_stats_0, HIERARCHY_CONFIGS[0]),
    (CLIENT_STATS, global_stats_1, HIERARCHY_CONFIGS[1]),
    (CLIENT_STATS, global_stats_2, HIERARCHY_CONFIGS[2]),
    (CLIENT_STATS, global_stats_3, HIERARCHY_CONFIGS[3]),
]


class TestHierarchicalNumericStats:
    def _round_global_stats(self, global_stats):
        if isinstance(global_stats, dict):
            for key, value in global_stats.items():
                if key == StC.GLOBAL or key == StC.LOCAL:
                    for key, metric in value.items():
                        if key == StC.STATS_HISTOGRAM:
                            for ds in metric:
                                for name, val in metric[ds].items():
                                    hist: Histogram = metric[ds][name]
                                    buckets = StatisticsController._apply_histogram_precision(hist.bins, 2)
                                    metric[ds][name] = buckets
                        else:
                            for ds in metric:
                                for name, val in metric[ds].items():
                                    metric[ds][name] = round(metric[ds][name], 2)
                    continue
                if isinstance(value, list):
                    for item in value:
                        self._round_global_stats(item)
        if isinstance(global_stats, list):
            for item in global_stats:
                self._round_global_stats(item)
        return global_stats

    @pytest.mark.parametrize("client_stats, expected_global_stats, hierarchy_configs", PARAMS)
    def test_global_stats(self, client_stats, expected_global_stats, hierarchy_configs):
        from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats

        global_stats = get_global_stats({}, client_stats, StC.STATS_1st_STATISTICS, hierarchy_configs)
        global_stats = get_global_stats(global_stats, client_stats, StC.STATS_2nd_STATISTICS, hierarchy_configs)
        result = self._round_global_stats(global_stats)

        assert expected_global_stats == result
