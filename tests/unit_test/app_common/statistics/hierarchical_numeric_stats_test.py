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
