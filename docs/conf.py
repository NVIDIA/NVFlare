# Copyright (c) 2021, NVIDIA CORPORATION.
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import subprocess
import sys
from pathlib import Path

import sphinx_rtd_theme
from sphinx.domains.python import PythonDomain
from sphinx.errors import ExtensionError


class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if "refspecific" in node:
            del node["refspecific"]
        return super(PatchedPythonDomain, self).resolve_xref(env, fromdocname, builder, typ, target, node, contnode)


sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "NVIDIA FLARE"
copyright = "2025 NVIDIA"
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
release = "2.7.0"
version = "2.7.0"

readthedocs_version_name = os.environ.get("READTHEDOCS_VERSION_NAME")
build_version = readthedocs_version_name if readthedocs_version_name not in (None, "latest", "stable") else "main"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Add napoleon to the extensions list
# source_parsers = {'.md': CommonMarkParser}

templates_path = ["templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

_skip_api = os.environ.get("SKIP_API_DOCS", "").lower() in ("1", "true", "yes")

extensions = [
    "recommonmark",
    "sphinx_llm.txt",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinxcontrib.jquery",
    "sphinx.ext.extlinks",
]

if not _skip_api:
    extensions.extend(
        [
            "sphinx.ext.autodoc",
            "sphinx.ext.viewcode",
        ]
    )

autoclass_content = "both"
add_module_names = False
autosectionlabel_prefix_document = True
llms_txt_description = (
    "NVIDIA FLARE is an open-source SDK for federated learning, with tools for simulation, job authoring, "
    "deployment, security, and production federated ML workflows."
)
llms_txt_full_build = False
llms_txt_build_parallel = False
llms_txt_suffix_mode = "replace"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

if _skip_api:
    # Exclude auto-generated API files but keep the committed stub
    # (docs/apidocs/modules.rst) so toctree / :doc: references still resolve.
    exclude_patterns.append("apidocs/nvflare*")

extlinks = {"github_nvflare_link": (f"https://github.com/NVIDIA/NVFlare/tree/{build_version}/%s", "")}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 1,
    "sticky_navigation": True,  # Set to False to disable the sticky nav while scrolling.
    # 'logo_only': True,  # if we have a html_logo below, this shows /only/ the logo with no title text
}
html_scaled_image_link = False
html_show_sourcelink = True
html_favicon = "favicon.ico"
html_logo = "resources/nvidia_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def generate_apidocs(*args):
    """Generate API docs automatically by trawling the available modules"""
    if os.environ.get("SKIP_API_DOCS", "").lower() in ("1", "true", "yes"):
        print("Skipping API doc generation (SKIP_API_DOCS is set)")
        return
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nvflare"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "apidocs"))
    print(f"output_path {output_path}")
    print(f"module_path {module_path}")
    subprocess.check_call(
        [sys.executable, "-m", "sphinx.ext.apidoc", "-f", "-e"]
        + ["-o", output_path]
        + [module_path]
        + [os.path.join(module_path, p) for p in exclude_patterns]
    )


def copy_curated_llms_txt(app, exception):
    """Publish the curated llms.txt after generated Markdown pages are copied."""
    if exception or not app.builder or app.builder.name not in ("html", "dirhtml"):
        return

    source_path = Path(app.confdir) / "llms.txt.in"
    target_path = Path(app.builder.outdir) / "llms.txt"
    llms_txt = source_path.read_text(encoding="utf-8")
    missing_links = []

    for link in re.findall(r"\]\(([^)]+)\)", llms_txt):
        if "://" in link or link.startswith("#"):
            continue
        linked_path = link.split("#", 1)[0]
        if linked_path and not (Path(app.builder.outdir) / linked_path).is_file():
            missing_links.append(link)

    if missing_links:
        raise ExtensionError(f"Missing generated Markdown files referenced by llms.txt: {missing_links}")

    target_path.write_text(llms_txt, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", generate_apidocs)
    app.connect("build-finished", copy_curated_llms_txt, priority=200)
    app.add_domain(PatchedPythonDomain, override=True)
    app.add_css_file("css/additions.css")
