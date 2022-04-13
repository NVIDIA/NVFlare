# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.workflows.analysis_controller import AnalysisController, SupportedTasks


class HistogramController(AnalysisController):
    def __init__(self, min_clients: int = 1, task_name_histogram: str = SupportedTasks.HISTOGRAM):
        """Controller for federated analysis.

        Args:
            min_clients: how many statistics to gather before computing the global statisitcs.
            task_name_histogram: task name for histogram computation.
        """
        super().__init__(
            min_clients=min_clients,
            task_name_histogram=task_name_histogram,
        )

    def visualize(self, fl_ctx: FLContext):
        global_histogram = None
        global_bin_edges = None
        global_n_images = 0
        global_n_included_images = 0

        n_clients = len(self.histograms.keys())
        # compute global histogram and plot local ones.
        if n_clients == 0:
            self.log_warning(fl_ctx, "There are no histograms!")
            return

        fig = make_subplots(rows=2, cols=n_clients)
        n_plots = 0

        i = 0
        for client_name, _histo in self.histograms.items():
            global_n_images += _histo.get("n_images", 0)
            global_n_included_images += _histo.get("n_included_images", 0)

            # compute global histogram
            if "histogram" in _histo:
                if global_histogram is None:
                    global_histogram = _histo["histogram"]
                    global_bin_edges = _histo["bin_edges"]
                else:  # add to current histogram
                    if np.all(np.equal(_histo["bin_edges"], global_bin_edges)):
                        global_histogram += _histo["histogram"]
                    else:
                        self.log_warning(
                            fl_ctx,
                            f"bin edges don't match the initial global bin edges. "
                            f"Ignoring results from {client_name} in global histogram.",
                        )

                # write local histogram to TensorBoard
                i += 1
                fig.add_trace(
                    go.Scatter(
                        x=_histo["bin_edges"],
                        y=_histo["histogram"],
                        name=f"{client_name} "
                        f"({_histo.get('n_included_images', 0)} of {_histo.get('n_images', 0)} images)",
                        mode="lines",
                        fill="tozeroy",
                    ),
                    row=1,
                    col=i,
                )
                n_plots += 1
        if global_histogram is not None:
            fig.add_trace(
                go.Scatter(
                    x=global_bin_edges,
                    y=global_histogram,
                    name=f"Global ({global_n_included_images} of {global_n_images} images)",
                    mode="lines",
                    fill="tozeroy",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            height=600,
            width=800,
            title_text=f"Histograms for {n_plots} of {n_clients} clients",
            xaxis_title="Image Intensity",
            yaxis_title="Count",
        )
        fig.write_html(os.path.join(self.run_dir, "histograms.html"))
        fig.write_image(os.path.join(self.run_dir, "histograms.svg"))
