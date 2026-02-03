#!/bin/bash
set -e

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

WORK_DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NUM_PARALLEL=1
DIR_TO_CHECK=("nvflare" "examples" "tests")

# Dependency caching settings
DEPS_MARKER="${WORK_DIR}/.deps_marker"
DEPS_MAX_AGE_DAYS=7
DEPS_MAX_RUNS=50

target="${@: -1}"
if [[ "${target}" == -* ]] ;then
    target=""
fi

function install_deps {
    if [[ $(uname) == "Darwin" ]]; then
      python3 -m pip install -e .[dev_mac]
    else
      python3 -m pip install -e .[dev]
    fi;
    echo "dependencies installed"
    # Reset the marker after successful install (atomic write)
    local tmp_marker="${DEPS_MARKER}.tmp.$$"
    printf "%s\n%s\n" "$(date +%s)" "0" > "$tmp_marker"
    mv "$tmp_marker" "$DEPS_MARKER"
}

function should_install_deps {
    # Returns 0 (true) if deps should be installed, 1 (false) otherwise
    # Marker file format: line 1 = install timestamp, line 2 = run count
    
    # If marker doesn't exist, install
    if [[ ! -f "$DEPS_MARKER" ]]; then
        echo "Dependencies not yet installed (no marker found)"
        return 0
    fi
    
    # Read marker file (line 1 = timestamp, line 2 = run count)
    local install_timestamp run_count
    install_timestamp=$(sed -n '1p' "$DEPS_MARKER" 2>/dev/null)
    run_count=$(sed -n '2p' "$DEPS_MARKER" 2>/dev/null)
    
    # Validate numeric values, default to 0 if empty or malformed
    [[ ! "$install_timestamp" =~ ^[0-9]+$ ]] && install_timestamp=0
    [[ ! "$run_count" =~ ^[0-9]+$ ]] && run_count=0
    
    # Check age (days since last install)
    local now marker_age_days
    now=$(date +%s)
    marker_age_days=$(( (now - install_timestamp) / 86400 ))
    
    if [[ $marker_age_days -ge $DEPS_MAX_AGE_DAYS ]]; then
        echo "Dependencies cache expired (${marker_age_days} days old, max ${DEPS_MAX_AGE_DAYS})"
        return 0
    fi
    
    # Check run count
    if [[ $run_count -ge $DEPS_MAX_RUNS ]]; then
        echo "Dependencies cache expired (${run_count} runs, max ${DEPS_MAX_RUNS})"
        return 0
    fi
    
    # Deps are cached and valid - increment run count (atomic write)
    echo "Dependencies cached (${run_count}/${DEPS_MAX_RUNS} runs, ${marker_age_days}/${DEPS_MAX_AGE_DAYS} days) - skipping install"
    local tmp_marker="${DEPS_MARKER}.tmp.$$"
    printf "%s\n%s\n" "$install_timestamp" "$((run_count + 1))" > "$tmp_marker"
    mv "$tmp_marker" "$DEPS_MARKER"
    return 1
}

function maybe_install_deps {
    if [[ "$fresh_deps" == "true" ]]; then
        echo "Forcing fresh dependency install..."
        install_deps
    elif should_install_deps; then
        install_deps
    fi
}

function clean {
    echo "remove coverage history"
    python3 -m coverage erase

    echo "uninstalling nvflare development files..."
    python3 setup.py develop --user --uninstall

    echo "removing temporary files in ${WORK_DIR}"

    find "${WORK_DIR}/nvflare" -type d -name "__pycache__" -exec rm -r "{}" \;
    find "${WORK_DIR}/nvflare" -type f -name "*.py[co]" -exec rm -r "{}" \;

    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name ".eggs" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name "nvflare.egg-info" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name "build" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name "dist" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name ".mypy_cache" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name ".pytype" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name ".coverage" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type f -name ".coverage.*" -exec rm -r "{}" \;
    find "${WORK_DIR}" -depth -maxdepth 1 -type d -name "__pycache__" -exec rm -r "{}" \;
    
    # Remove dependency cache marker
    rm -f "$DEPS_MARKER"
    echo "dependency cache cleared"
}

function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}

function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
    echo "Please run auto style fixes: ${green}./runtest.sh -f${noColor}"
}
function report_status() {
    status="$1"
    if [ "${status}" -ne 0 ]
    then
        print_style_fail_msg
        exit "${status}"
    else
        echo "${green}passed!${noColor}"
    fi
}

function is_pip_installed() {
    echo "check target installed by pip"
    return $(python3 -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader(sys.argv[1]) else 1)" $1)
}

function dry_run() {
    echo "${separator}${blue}dryrun${noColor}"
    echo "    " "$1"
}

function check_license() {
    folders_to_check_license="nvflare examples tests integration research"
    echo "checking license header in folder: $folders_to_check_license"
    (grep -r --include "*.py" --exclude-dir "*protos*" --exclude "modeling_roberta.py" -L \
    "\(# Copyright (c) \(2021\|2022\|2023\|2024\|2025\|2026\), NVIDIA CORPORATION.  All rights reserved.\)\|\(This file is released into the public domain.\)" \
    ${folders_to_check_license} || true) > no_license.lst
    if [ -s no_license.lst ]; then
        # The file is not-empty.
        cat no_license.lst
        echo "License text not found on the above files."
        echo "Please fix them."
        rm -f no_license.lst
        exit 1
    else
        echo "All Python files in folder ${folders_to_check_license} have license header"
        rm -f no_license.lst
    fi
    echo "finished checking license header"
}

function flake8_check() {
    echo "${separator}${blue}flake8${noColor}"
    python3 -m flake8 --version
    python3 -m flake8 "$@" --count --statistics
    report_status "$?"
}

function black_check() {
    echo "${separator}${blue}black-check${noColor}"
    python3 -m black --check "$@"
    report_status "$?"
    echo "Done with black code style checks on $@"
}

function black_fix() {
    echo "${separator}${blue}black-fix${noColor}"
    python3 -m black "$@"
    echo "Done with black code style fix on $@"
}

function isort_check() {
    echo "${separator}${blue}isort-check${noColor}"
    python3 -m isort --check "$@"
    report_status "$?"
}

function isort_fix() {
    echo "${separator}${blue}isort-fix${noColor}"
    python3 -m isort "$@"
}

# wait for approval
#function pylint_check() {
#        echo "${separator}${blue}pylint${noColor}"
#        python3 -m pylint --version
#        ignore_codes="E1101,E1102,E0601,E1130,E1123,E0102,E1120,E1137,E1136"
#        python3 -m pylint "$1" -E --disable=$ignore_codes -j $NUM_PARALLEL
#        report_status "$?"
#}

function pytype_check() {
    echo "${separator}${blue}pytype${noColor}"
    pytype_ver=$(python3 -m pytype --version)
    if [[ "$OSTYPE" == "darwin"* && "$pytype_ver" == "2021."* ]]; then
        echo "${red}pytype not working on macOS 2021 (https://github.com/google/pytype/issues/661). Please upgrade to 2022*.${noColor}"
        exit 1
    else
        python3 -m pytype --version
        python3 -m pytype -j ${NUM_PARALLEL} --python-version="$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")" "$@"
        report_status "$?"
    fi
}

function mypy_check() {
    echo "${separator}${blue}mypy${noColor}"
    python3 -m mypy --version
    python3 -m mypy "$1"
    report_status "$?"
}

function check_style_type_import() {
    # remove pylint for now
    # pylint_check  "$@"
    black_check   "$@"
    isort_check   "$@"
    flake8_check  "$@"
    # pytype causing check fails, comment for now
    # pytype_check  "$@"

    # gives a lot false alarm, comment out for now
    # mypy_check    "$@"
}

function fix_style_import() {
    black_fix "$@"
    isort_fix "$@"
}

function get_current_kernel() {
    # Prefer a stable kernelspec name if it exists.
    if command -v jupyter >/dev/null 2>&1; then
        if jupyter kernelspec list 2>/dev/null | awk '{print $1}' | grep -qx "python3"; then
            echo "python3"
            return
        fi
    fi

    # Otherwise, don't guess. Let nbmake use the notebook's kernelspec metadata.
    echo ""
}

function validate_kernel() {
    local kernel_name="$1"
    if ! command -v jupyter >/dev/null 2>&1; then
        echo "${red}Error: jupyter not found. Install with: pip install jupyter${noColor}"
        return 1
    fi
    if ! jupyter kernelspec list 2>/dev/null | awk '{print $1}' | grep -qx "$kernel_name"; then
        echo "${red}Error: kernel '$kernel_name' not found${noColor}"
        echo "Available kernels:"
        jupyter kernelspec list 2>/dev/null | tail -n +2
        return 1
    fi
    return 0
}

function notebook_test() {
    echo "${separator}${blue}notebook-test${noColor}"

    # Auto-detect kernel at runtime if not specified (allows jupyter to be installed after script load)
    local kernel="${nb_kernel:-$(get_current_kernel)}"

    if [[ -n "$kernel" ]]; then
        if ! validate_kernel "$kernel"; then
            exit 1
        fi
        echo "Using kernel: $kernel"
    else
        echo "No kernel specified, using notebook's default kernel"
    fi

    echo "Timeout: ${nb_timeout}s, Clean mode: ${nb_clean}"

    # Build and print the exact command for debugging
    local cmd
    cmd=(python3 -m pytest --nbmake --nbmake-timeout="$nb_timeout" --nbmake-clean="$nb_clean")
    if [[ -n "$kernel" ]]; then
        cmd+=("--kernel=$kernel")
    fi
    cmd+=("$@")
    if [[ "$verbose_flag" == "true" ]]; then
        cmd+=(-v)
    fi

    echo "Executing: ${cmd[*]}"
    "${cmd[@]}"
    report_status "$?"
}

################################################################################
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH} && echo "PYTHONPATH is ${PYTHONPATH}"

function help() {
    echo "Add description of the script functions here."
    echo
    echo "Syntax: runtest.sh  [-h|--help]
                               [-l|--check-license]
                               [-s|--check-format]
                               [-f|--fix-format]
                               [-u|--unit-tests]
                               [-n|--notebook]
                               [-r|--test-report]
                               [-c|--coverage]
                               <target> "
    echo "<target> : target directories or files used for unit tests, or license check or format checks etc. "
    echo " "
    echo "options:"
    echo ""
    echo "    -h | --help                   : display usage help"
    echo "    -l | --check-license          : check copy license"
    echo "    -s | --check-format           : check code styles, formatting, typing, import"
    echo "    -f | --fix-format             : auto fix style formats, import"
    echo "    -u | --unit-tests             : unit tests"
    echo "    -n | --notebook               : notebook tests using nbmake"
    echo "    -r | --test-report            : used with -u command, turn on unit test report flag. It has no effect without -u "
    echo "    -p | --dependencies           : only install dependencies"
    echo "    -c | --coverage               : used with -u command, turn on coverage flag,  It has no effect without -u "
    echo "    -v | --verbose                : verbose output (adds -v to pytest)"
    echo "    -d | --dry-run                : set dry run flag, print out command"
    echo "         --fresh-deps             : force fresh dependency install (bypasses cache)"
    echo "         --clean                  : clean py and other artifacts generated, clears dependency cache"
#   echo "    -i | --integration-tests      : integration tests"
    echo ""
    echo "Notebook test options (used with -n):"
    echo "         --timeout=SECONDS        : timeout per notebook (default: 1200)"
    echo "         --kernel=NAME            : Jupyter kernel name (must be valid, see: jupyter kernelspec list)"
    echo "         --nb-clean=MODE          : clean outputs: always, on-success, never (default: on-success)"
    echo ""
    echo "Examples:"
    echo "    ./runtest.sh -n                                              # test default (flare_simulator.ipynb)"
    echo "    ./runtest.sh -n -v examples/tutorials/flare_api.ipynb        # test single notebook with verbose"
    echo "    ./runtest.sh -n --timeout=1800 --kernel=python3 examples/tutorials/"
    exit 1
}

coverage_report=false
unit_test_report=false
dry_run_flag=false
verbose_flag=false
fresh_deps=false

# notebook test defaults
nb_timeout=1200
nb_clean="on-success"
nb_kernel=""  # Auto-detected at runtime if not specified via --kernel=

# parse arguments
cmd=""

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            help
            exit
        ;;

        -l|--check-license)
            cmd="check_license"
        ;;

        -s |--check-format) # check format and styles
            cmd="check_style_type_import"
            if [[ -z $target ]]; then
                target="${DIR_TO_CHECK[@]}"
            fi
        ;;

        -f |--fix-format)
            cmd="fix_style_import"
            if [[ -z $target ]]; then
                target="${DIR_TO_CHECK[@]}"
            fi
        ;;
        -c|--coverage)
            coverage_report=true
        ;;

        -p|--dependencies)
            dependencies=true
            cmd=" "
        ;;

        -r|--test-report)
            echo "set unit test flag"
            unit_test_report=true
        ;;

        -u |--unit*)
            cmd_prefix="python3 -m pytest --numprocesses=8 -v "

            echo "coverage_report=" ${coverage_report}
            if [ "${coverage_report}" == true ]; then
                cmd_prefix="${cmd_prefix} --cov=${target} --cov-report html:cov_html --cov-report xml:cov.xml"
            fi

            if [ "${unit_test_report}" == true ]; then
                cmd_prefix="${cmd_prefix} --junitxml=unit_test.xml "
            fi
            cmd="$cmd_prefix"

            if [[ -z $target ]]; then
                target="tests/unit_test"
            fi
        ;;

        -n|--notebook)
            cmd="notebook_test"
            if [[ -z $target ]]; then
                target="examples/tutorials/flare_simulator.ipynb"
            fi
        ;;

        --timeout=*)
            nb_timeout="${key#*=}"
        ;;

        --kernel=*)
            nb_kernel="${key#*=}"
        ;;

        --nb-clean=*)
            nb_clean="${key#*=}"
        ;;

        --clean)
            cmd="clean"
        ;;

        -d|--dry-run)
            dry_run_flag=true
        ;;

        -v|--verbose)
            verbose_flag=true
        ;;

        --fresh-deps)
            fresh_deps=true
        ;;

        -*)
            help
        ;;

    esac
    shift
done

if [[ -z $cmd ]]; then
    cmd="check_license;
        check_style_type_import "${DIR_TO_CHECK[@]}";
        fix_style_import "${DIR_TO_CHECK[@]}";
        python3 -m pytest --numprocesses=8 -v --cov=nvflare --cov-report html:cov_html --cov-report xml:cov.xml --junitxml=unit_test.xml --dist loadgroup tests/unit_test;
        "
else
    cmd="$cmd $target"
fi

echo "running command: "
echo "        $cmd"
echo "                 "
if [[ $dry_run_flag = "true" ]]; then
    dry_run "$cmd"
else
    maybe_install_deps
    eval "$cmd"
fi
echo "Done"
