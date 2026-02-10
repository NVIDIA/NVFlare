#!/bin/bash
set -euo pipefail

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

function print_usage() {
    echo "build_doc.sh [--clean] [--html] [--skip-api]"
    echo ""
    echo "Build documentation"
    echo ""
    echo "Options:"
    echo "  --html        Build HTML docs (full build including API reference)."
    echo "  --skip-api    Skip API reference generation (faster dev builds)."
    echo "                Sets SKIP_API_DOCS=1 environment variable."
    echo "  --clean       Clean up build artifacts."
    echo ""
    echo "Examples:"
    echo "./build_doc.sh --html              # full production build with API reference."
    echo "./build_doc.sh --html --skip-api   # fast dev build, skip API reference."
    echo "./build_doc.sh --clean             # clean up python build related files."
    echo ""
    echo "You can also set the env var directly:"
    echo "SKIP_API_DOCS=1 ./build_doc.sh --html"
}

function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}

function clean_py() {
    find nvflare -type f -name "*.py[co]" -delete
    find nvflare -type f -name "*.so" -delete
    find nvflare -type d -name "__pycache__" -exec rm -r "{}" +

    find . -depth -maxdepth 1 -type d -name ".eggs" -exec rm -r "{}" +
    find . -depth -maxdepth 1 -type d -name "*.egg-info" -exec rm -r "{}" +
    find . -depth -maxdepth 1 -type d -name "build" -exec rm -r "{}" +
    find . -depth -maxdepth 1 -type d -name "dist" -exec rm -r "{}" +
    find . -depth -maxdepth 1 -type d -name "__pycache__" -exec rm -r "{}" +
}

function clean_docs() {
    find docs -type d -name "_build" -exec rm -r "{}" +
    find docs/apidocs -type f -name "nvflare*" -delete
}

function build_html_docs() {
    pip install -e .[dev]
    if [[ "$SKIP_API_DOCS" == "1" ]]; then
        echo "${blue}Skipping API reference generation (SKIP_API_DOCS=1)${noColor}"
    else
        echo "${blue}Generating API reference...${noColor}"
        sphinx-apidoc --module-first -f -o docs/apidocs/ nvflare "*poc" "*private"
    fi
    sphinx-build -b html docs docs/_build
}

if [ -z "$1" ]
then
    print_error_msg "Too few arguments to $0"
    print_usage
fi

# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --clean)
            doClean=true
        ;;
        --html)
            doHTML=true
        ;;
        --skip-api)
            export SKIP_API_DOCS=1
        ;;
        *)
            print_error_msg "Incorrect commandline provided, invalid key: $key"
            print_usage
        ;;
    esac
    shift
done

if [[ $doClean == true ]]
then
    echo "${separator}${blue}clean${noColor}"
    clean_py
    clean_docs
    echo "${green}done!${noColor}"
    exit
fi

if [[ $doHTML == true ]]
then
    echo "${separator}${blue}html${noColor}"
    clean_docs
    build_html_docs
    echo "${green}done!${noColor}"
    exit
fi
