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

import itertools
from math import isnan, sqrt

import numpy as np
import pandas as pd
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                        "var": {"data_set1": {"Feature1": 0.01}},
                                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                        "var": {"data_set1": {"Feature1": 0.01}},
                                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                        "var": {"data_set1": {"Feature1": 0.01}},
                                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                        "var": {"data_set1": {"Feature1": 0.01}},
                                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                                "var": {"data_set1": {"Feature1": 0.01}},
                                "stddev": {"data_set1": {"Feature1": 0.1}},
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
                        "var": {"data_set1": {"Feature1": 0.01}},
                        "stddev": {"data_set1": {"Feature1": 0.1}},
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
                "var": {"data_set1": {"Feature1": 0.01}},
                "stddev": {"data_set1": {"Feature1": 0.1}},
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

STATES_CONFIG = {
    "States": [{"Name": "State-A", "Sites": ["Site-1", "Site-2"]}, {"Name": "State-B", "Sites": ["Site-3"]}]
}


def _two_round_stats(site_data: dict, hierarchy_config: dict, precision: int = 4) -> dict:
    """Run both statistics rounds the way the controller does, with client values computed like DFStatisticsCore."""
    from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats

    series = {site: pd.Series(values, dtype=float) for site, values in site_data.items()}
    client_stats = {StC.STATS_COUNT: {}, StC.STATS_SUM: {}, StC.STATS_MEAN: {}}
    for site, s in series.items():
        client_stats[StC.STATS_COUNT][site] = {"ds": {"f": s.count()}}
        client_stats[StC.STATS_SUM][site] = {"ds": {"f": round(s.sum(), precision)}}
        client_stats[StC.STATS_MEAN][site] = {"ds": {"f": round(s.mean(), precision)}}
    global_stats = get_global_stats({}, client_stats, StC.STATS_1st_STATISTICS, hierarchy_config)

    global_mean = global_stats[StC.GLOBAL][StC.STATS_MEAN]["ds"]["f"]
    global_count = global_stats[StC.GLOBAL][StC.STATS_COUNT]["ds"]["f"]
    client_stats[StC.STATS_VAR] = {}
    client_stats[StC.STATS_STDDEV] = {}
    for site, s in series.items():
        var = ((s - global_mean) * (s - global_mean)).sum() / (global_count - 1)
        client_stats[StC.STATS_VAR][site] = {"ds": {"f": round(var, precision)}}
        client_stats[StC.STATS_STDDEV][site] = {"ds": {"f": round(s.std(), precision)}}
    return get_global_stats(global_stats, client_stats, StC.STATS_2nd_STATISTICS, hierarchy_config)


def _vars_and_stddevs(global_stats: dict) -> dict:
    result = {StC.GLOBAL: global_stats[StC.GLOBAL]}
    for state in global_stats["States"]:
        result[state[StC.NAME]] = state[StC.GLOBAL]
        for site in state["Sites"]:
            result[site[StC.NAME]] = site[StC.LOCAL]
    return {k: (v[StC.STATS_VAR]["ds"]["f"], v[StC.STATS_STDDEV]["ds"]["f"]) for k, v in result.items()}


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

    def test_var_and_stddev_of_every_level(self):
        data = {"Site-1": [1, 2, 3, 4], "Site-2": [10, 11, 12, 13], "Site-3": [100, 101, 102, 103]}
        result = _vars_and_stddevs(_two_round_stats(data, STATES_CONFIG))

        expected = {
            StC.GLOBAL: 2181.0,
            "State-A": 24.5714,
            "State-B": 1.6667,
            "Site-1": 1.6667,
            "Site-2": 1.6667,
            "Site-3": 1.6667,
        }
        assert result.keys() == expected.keys()
        for level, var in expected.items():
            assert result[level][0] == pytest.approx(var, abs=5e-4), level
            assert result[level][1] == pytest.approx(sqrt(var), abs=5e-4), level

    def test_var_matches_numpy_with_uneven_sites(self):
        rng = np.random.default_rng(0)
        data = {
            "Site-1": rng.normal(10, 2, 50).tolist(),
            "Site-2": rng.normal(-5, 7, 3000).tolist(),
            "Site-3": rng.uniform(0, 1, 30).tolist(),
        }
        result = _vars_and_stddevs(_two_round_stats(data, STATES_CONFIG))

        records = {
            StC.GLOBAL: data["Site-1"] + data["Site-2"] + data["Site-3"],
            "State-A": data["Site-1"] + data["Site-2"],
            "State-B": data["Site-3"],
            **data,
        }
        for level, values in records.items():
            # Clients round their stddev to 4 digits, which moves the var by about 2 * stddev * 5e-5 at most
            var = np.var(values, ddof=1)
            assert result[level][0] == pytest.approx(var, abs=1e-3), level
            assert result[level][1] == pytest.approx(sqrt(var), abs=1e-4), level

    def test_single_record_site(self):
        # pandas returns NaN for the stddev of a single record, so the site adds nothing to the within-site term
        data = {"Site-1": [1, 2, 3, 4], "Site-2": [10], "Site-3": [100]}
        result = _vars_and_stddevs(_two_round_stats(data, STATES_CONFIG))

        assert result[StC.GLOBAL][0] == pytest.approx(np.var([1, 2, 3, 4, 10, 100], ddof=1), abs=1e-4)
        assert result["State-A"][0] == pytest.approx(np.var([1, 2, 3, 4, 10], ddof=1), abs=1e-4)
        assert result["Site-1"][0] == pytest.approx(np.var([1, 2, 3, 4], ddof=1), abs=1e-4)
        assert isnan(result["Site-2"][0]) and isnan(result["Site-2"][1])
        assert isnan(result["State-B"][0]) and isnan(result["State-B"][1])

    def test_empty_site(self):
        data = {"Site-1": [1, 2, 3, 4], "Site-2": [], "Site-3": [100, 101, 102, 103]}
        with np.errstate(invalid="ignore"):
            result = _vars_and_stddevs(_two_round_stats(data, STATES_CONFIG))

        assert result[StC.GLOBAL][0] == pytest.approx(np.var([1, 2, 3, 4, 100, 101, 102, 103], ddof=1), abs=1e-4)
        assert result["State-A"][0] == pytest.approx(np.var([1, 2, 3, 4], ddof=1), abs=1e-4)
        assert result["Site-1"][0] == pytest.approx(np.var([1, 2, 3, 4], ddof=1), abs=1e-4)
        assert isnan(result["Site-2"][0])
        assert result["State-B"][0] == pytest.approx(np.var([100, 101, 102, 103], ddof=1), abs=1e-4)

    def test_var_without_client_stddev_fails(self):
        from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats

        client_stats = {k: v for k, v in CLIENT_STATS.items() if k != StC.STATS_STDDEV}
        global_stats = get_global_stats({}, client_stats, StC.STATS_1st_STATISTICS, HIERARCHY_CONFIGS[1])
        with pytest.raises(ValueError, match="stddev"):
            get_global_stats(global_stats, client_stats, StC.STATS_2nd_STATISTICS, HIERARCHY_CONFIGS[1])


class TestHierarchicalWithheldSites:
    HIERARCHY = {
        "States": [
            {"Name": "State-A", "Sites": ["Site-1", "Site-2"]},
            {"Name": "State-B", "Sites": ["Site-3"]},
            {"Name": "State-C", "Sites": ["Site-4"]},
        ]
    }
    # Site-1 and Site-4 are below the privacy filter's min_count, so their statistics are withheld.
    DATA = {
        "Site-1": [11.0, 12.0, 15.0],
        "Site-2": list(9.0 + np.linspace(-1.0, 1.0, 40)),
        "Site-3": list(11.0 + np.linspace(-2.0, 2.0, 60)),
        "Site-4": [4.0, 5.0],
    }

    def _run(self, order):
        """Run both rounds through the real MinCountCleanser, as the privacy filter and controller do."""
        from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats
        from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser

        cleanser = MinCountCleanser(min_count=10)
        withheld = {}

        def collect(site, statistics):
            for ds_name, features in cleanser.withheld_features(statistics, site).items():
                withheld.setdefault(site, {}).setdefault(ds_name, set()).update(features)
            return cleanser.apply(statistics, site)[0]

        series = {site: pd.Series(self.DATA[site], dtype=float) for site in order}
        first = {}
        for site, s in series.items():
            first[site] = collect(
                site,
                {
                    StC.STATS_COUNT: {"ds": {"f": float(s.count())}},
                    StC.STATS_FAILURE_COUNT: {"ds": {"f": 0.0}},
                    StC.STATS_SUM: {"ds": {"f": float(s.sum())}},
                    StC.STATS_MEAN: {"ds": {"f": float(s.mean())}},
                },
            )
        metrics = {m: {site: first[site][m] for site in order} for m in first[order[0]]}
        global_stats = get_global_stats({}, metrics, StC.STATS_1st_STATISTICS, self.HIERARCHY, withheld)

        global_mean = global_stats[StC.GLOBAL][StC.STATS_MEAN]["ds"]["f"]
        global_count = global_stats[StC.GLOBAL][StC.STATS_COUNT]["ds"]["f"]
        second = {}
        for site, s in series.items():
            bins = [Bin(0.0, 10.0, int((s < 10).sum())), Bin(10.0, 20.0, int((s >= 10).sum()))]
            second[site] = collect(
                site,
                {
                    StC.STATS_COUNT: {"ds": {"f": float(s.count())}},
                    StC.STATS_FAILURE_COUNT: {"ds": {"f": 0.0}},
                    StC.STATS_HISTOGRAM: {"ds": {"f": Histogram(HistogramType.STANDARD, bins)}},
                    StC.STATS_VAR: {"ds": {"f": float(((s - global_mean) ** 2).sum() / (global_count - 1))}},
                    StC.STATS_STDDEV: {"ds": {"f": float(s.std())}},
                },
            )
        metrics = {m: {site: second[site][m] for site in order} for m in second[order[0]]}
        return get_global_stats(global_stats, metrics, StC.STATS_2nd_STATISTICS, self.HIERARCHY, withheld)

    @staticmethod
    def _levels(result):
        levels = {"Global": result[StC.GLOBAL]}
        for state in result["States"]:
            levels[state[StC.NAME]] = state[StC.GLOBAL]
            for site in state["Sites"]:
                levels[site[StC.NAME]] = site[StC.LOCAL]
        return levels

    @pytest.mark.parametrize("order", list(itertools.permutations(["Site-1", "Site-2", "Site-3", "Site-4"])))
    def test_withheld_sites_do_not_affect_the_aggregate_levels(self, order):
        levels = self._levels(self._run(list(order)))
        contributing = np.array(self.DATA["Site-2"] + self.DATA["Site-3"])

        assert levels["Global"][StC.STATS_COUNT]["ds"]["f"] == 100
        assert levels["Global"][StC.STATS_MEAN]["ds"]["f"] == pytest.approx(contributing.mean())
        assert levels["Global"][StC.STATS_VAR]["ds"]["f"] == pytest.approx(contributing.var(ddof=1), abs=1e-6)
        for state, site in (("State-A", "Site-2"), ("State-B", "Site-3")):
            values = np.array(self.DATA[site])
            assert levels[state][StC.STATS_COUNT]["ds"]["f"] == len(values)
            assert levels[state][StC.STATS_MEAN]["ds"]["f"] == pytest.approx(values.mean())
            assert levels[state][StC.STATS_VAR]["ds"]["f"] == pytest.approx(values.var(ddof=1), abs=1e-6)
            assert levels[site][StC.STATS_VAR]["ds"]["f"] == pytest.approx(values.var(ddof=1), abs=1e-6)

    def test_level_without_contributing_sites_has_no_entry_for_the_feature(self):
        state_c = self._levels(self._run(["Site-1", "Site-2", "Site-3", "Site-4"]))["State-C"]

        assert all("f" not in values["ds"] for values in state_c.values())

    def test_withheld_site_has_no_local_entry_for_the_feature(self):
        levels = self._levels(self._run(["Site-1", "Site-2", "Site-3", "Site-4"]))

        for site in ("Site-1", "Site-4"):
            assert all("f" not in values["ds"] for values in levels[site].values())
        assert levels["Site-2"][StC.STATS_COUNT]["ds"]["f"] == 40

    def test_result_can_be_rounded_by_the_controller(self):
        from nvflare.app_common.workflows.hierarchical_statistics_controller import HierarchicalStatisticsController

        result = self._run(["Site-4", "Site-2", "Site-1", "Site-3"])
        controller = HierarchicalStatisticsController(
            statistic_configs={StC.STATS_COUNT: {}}, writer_id="stats_writer", hierarchy_config="hierarchy_config.json"
        )

        controller._recursively_round_global_stats(result)

        # The controller replaces each Histogram with its list of rounded bins.
        global_bins = result[StC.GLOBAL][StC.STATS_HISTOGRAM]["ds"]["f"]
        assert sum(b.sample_count for b in global_bins) == 100


class TestRemoveUnreportedFromOutput:
    @staticmethod
    def _entry(features=("f", "g")):
        return {StC.STATS_COUNT: {"ds": {name: 0 for name in features}}}

    def _tree(self):
        def site(name):
            return {StC.NAME: name, StC.LOCAL: self._entry()}

        def level(name, key, children):
            return {StC.NAME: name, StC.GLOBAL: self._entry(), key: children}

        return {
            StC.GLOBAL: self._entry(),
            "Regions": [
                level(
                    "R1",
                    "States",
                    [level("S1", "Sites", [site("a"), site("b")]), level("S2", "Sites", [site("c")])],
                )
            ],
        }

    def test_nested_levels_keep_features_that_some_site_reported(self):
        from nvflare.app_common.statistics.hierarchical_numeric_stats import remove_unreported_from_output

        tree = self._tree()
        metrics = {"count": {"a": {"ds": {"g": 10}}, "b": {"ds": {"f": 20}}, "c": {"ds": {}}}}

        sites = remove_unreported_from_output(tree, metrics, ["count"])

        assert sites == ["a", "b", "c"]
        region = tree["Regions"][0]
        state_1, state_2 = region["States"]
        features = lambda node: set(node["count"]["ds"])  # noqa: E731
        assert features(tree[StC.GLOBAL]) == {"f", "g"}
        assert features(region[StC.GLOBAL]) == {"f", "g"}
        assert features(state_1[StC.GLOBAL]) == {"f", "g"}  # a reported g, b reported f
        assert features(state_2[StC.GLOBAL]) == set()  # its only site reported neither
        assert features(state_1["Sites"][0][StC.LOCAL]) == {"g"}
        assert features(state_1["Sites"][1][StC.LOCAL]) == {"f"}
        assert features(state_2["Sites"][0][StC.LOCAL]) == set()

    def test_top_level_loses_a_feature_only_when_no_site_reported_it(self):
        from nvflare.app_common.statistics.hierarchical_numeric_stats import remove_unreported_from_output

        tree = self._tree()
        metrics = {"count": {name: {"ds": {"g": 10}} for name in ("a", "b", "c")}}

        remove_unreported_from_output(tree, metrics, ["count"])

        assert set(tree[StC.GLOBAL]["count"]["ds"]) == {"g"}
        assert set(tree["Regions"][0][StC.GLOBAL]["count"]["ds"]) == {"g"}


class TestHierarchicalWithheldInSecondRound:
    HIERARCHY = {"States": [{"Name": "State-A", "Sites": ["Site-1", "Site-2"]}]}

    def _rounds(self, failures_in_second_round):
        from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats
        from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser

        cleanser = MinCountCleanser(min_count=10)
        withheld = {}

        def collect(site, statistics):
            for ds_name, features in cleanser.withheld_features(statistics, site).items():
                withheld.setdefault(site, {}).setdefault(ds_name, set()).update(features)
            return cleanser.apply(statistics, site)[0]

        values = {"Site-1": 1.0, "Site-2": 3.0}
        first = {
            site: collect(
                site,
                {
                    StC.STATS_COUNT: {"ds": {"f": 10.0}},
                    StC.STATS_FAILURE_COUNT: {"ds": {"f": 0.0}},
                    StC.STATS_SUM: {"ds": {"f": 10 * v}},
                    StC.STATS_MEAN: {"ds": {"f": v}},
                },
            )
            for site, v in values.items()
        }
        metrics = {m: {site: first[site][m] for site in values} for m in first["Site-1"]}
        global_stats = get_global_stats({}, metrics, StC.STATS_1st_STATISTICS, self.HIERARCHY, withheld)

        histogram = Histogram(HistogramType.STANDARD, [Bin(0.0, 5.0, 10)])
        second = {
            site: collect(
                site,
                {
                    StC.STATS_COUNT: {"ds": {"f": 10.0}},
                    StC.STATS_FAILURE_COUNT: {"ds": {"f": failures_in_second_round.get(site, 0.0)}},
                    StC.STATS_HISTOGRAM: {"ds": {"f": histogram}},
                },
            )
            for site in values
        }
        metrics = {m: {site: second[site][m] for site in values} for m in second["Site-1"]}
        return get_global_stats(global_stats, metrics, StC.STATS_2nd_STATISTICS, self.HIERARCHY, withheld)

    def test_site_withheld_only_in_the_second_round_is_rejected(self):
        with pytest.raises(ValueError, match="Site-2.*2nd statistics round"):
            self._rounds({"Site-2": 1.0})

    def test_no_error_when_no_site_changes_between_rounds(self):
        result = self._rounds({})

        assert result[StC.GLOBAL][StC.STATS_COUNT]["ds"]["f"] == 20
        assert result[StC.GLOBAL][StC.STATS_HISTOGRAM]["ds"]["f"].bins[0].sample_count == 20


class TestHierarchicalMixedFeatures:
    HIERARCHY = {
        "Regions": [
            {
                "Name": "Region",
                "States": [
                    {"Name": "State-A", "Sites": ["a", "b"]},
                    {"Name": "State-B", "Sites": ["c"]},
                ],
            }
        ]
    }

    @pytest.mark.parametrize("order", list(itertools.permutations(["a", "b", "c"])))
    @pytest.mark.parametrize("withhold", [False, True])
    @pytest.mark.parametrize("derived", [False, True], ids=["histogram-only", "mean-and-variance"])
    @pytest.mark.parametrize("different_dataset", [False, True])
    def test_only_reported_features_appear_at_each_level(self, order, withhold, derived, different_dataset):
        from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats
        from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser
        from nvflare.app_common.workflows.hierarchical_statistics_controller import HierarchicalStatisticsController

        # A withholds f, B does not have f at all, and C reports f. B can also lack A's entire dataset.
        data = {
            "a": {"train": {"f": np.array([2.0, 3.0, 4.0])}},
            "b": {"other" if different_dataset else "train": {"g": np.arange(12.0) + 10}},
            "c": {"train": {"f": np.arange(20.0) + 20, "g": np.arange(22.0) + 30}},
        }
        cleanser = MinCountCleanser(min_count=10 if withhold else 0)
        withheld = {}
        metrics = {}
        result = {}
        for task in (StC.STATS_1st_STATISTICS, StC.STATS_2nd_STATISTICS):
            first = task == StC.STATS_1st_STATISTICS
            names = ["count", "failure_count"]
            names += (["sum", "mean"] if first else ["var", "stddev"]) if derived else []
            names += ["min", "max"] if first else ["histogram"]
            for site in order:
                statistics = {m: {ds: {} for ds in data[site]} for m in names}
                for ds, features in data[site].items():
                    for feature, values in features.items():
                        statistics["count"][ds][feature] = len(values)
                        statistics["failure_count"][ds][feature] = 0
                        if first:
                            statistics["min"][ds][feature] = float(values.min())
                            statistics["max"][ds][feature] = float(values.max())
                            if derived:
                                statistics["sum"][ds][feature] = float(values.sum())
                                statistics["mean"][ds][feature] = float(values.mean())
                        else:
                            statistics["histogram"][ds][feature] = Histogram(
                                HistogramType.STANDARD, [Bin(0.0, 100.0, len(values))]
                            )
                            if derived:
                                top = result[StC.GLOBAL]
                                count = top["count"][ds].get(feature, 0)
                                mean = top["mean"][ds].get(feature, 0)
                                statistics["var"][ds][feature] = (
                                    float(((values - mean) ** 2).sum() / (count - 1)) if count > 1 else None
                                )
                                statistics["stddev"][ds][feature] = float(values.std(ddof=1))
                for ds, features in cleanser.withheld_features(statistics, site).items():
                    withheld.setdefault(site, {}).setdefault(ds, set()).update(features)
                for metric, values in cleanser.apply(statistics, site)[0].items():
                    metrics.setdefault(metric, {})[site] = values
            result = get_global_stats(result, metrics, task, self.HIERARCHY, withheld)

        def verify(node):
            sites = []
            for value in node.values():
                if isinstance(value, list):
                    for child in value:
                        sites.extend(verify(child))
            if StC.LOCAL in node:
                sites = [node[StC.NAME]]
            local = node.get(StC.LOCAL, node.get(StC.GLOBAL))
            for ds in local["count"]:
                reported = {}
                for site in sites:
                    for feature, values in data[site].get(ds, {}).items():
                        if feature not in withheld.get(site, {}).get(ds, set()):
                            reported.setdefault(feature, []).append(values)
                for metric in local.values():
                    assert set(metric[ds]) == set(reported)
                for feature, parts in reported.items():
                    values = np.concatenate(parts)
                    assert local["count"][ds][feature] == len(values)
                    assert local["min"][ds][feature] == values.min()
                    assert local["max"][ds][feature] == values.max()
                    assert sum(b.sample_count for b in local["histogram"][ds][feature].bins) == len(values)
                    if derived:
                        assert local["mean"][ds][feature] == pytest.approx(values.mean())
                        assert local["var"][ds][feature] == pytest.approx(values.var(ddof=1))
                        assert local["stddev"][ds][feature] == pytest.approx(values.std(ddof=1))
            return sites

        verify(result)
        controller = HierarchicalStatisticsController({"histogram": {}}, "writer", hierarchy_config="hierarchy.json")
        controller._recursively_round_global_stats(result)
