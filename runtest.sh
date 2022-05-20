#!/usr/bin/env bash 
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

target="${@: -1}"
if [[ "${target}" == -* ]] ;then
    target=""
fi


function install_deps {
    if [[ ! -f /tmp/.flare_deps_installed ]]; then
      echo "pip installing development dependencies"
      python3 -m pip install -r requirements-dev.txt
      echo "dependencies installed" > /tmp/.flare_deps_installed
    fi;
}

function clean {
    echo "remove flare_deps_installed flag"
    if [[ -f /tmp/.flare_deps_installed ]]; then
       rm -r /tmp/.flare_deps_installed
    fi;

    echo "remove coverage history"
    python3 -m coverage erase

    echo "uninstalling nvflare development files..."
    python3 setup.py develop --user --uninstall

    echo "removing temporary files in ${WORK_DIR}"

    find ${WORK_DIR}/nvflare -type d -name "__pycache__" -exec rm -r "{}" \;
    find ${WORK_DIR}/nvflare -type f -name "*.py[co]" -exec rm -r "{}" \;

    find ${WORK_DIR} -depth -maxdepth 1 -type d -name ".eggs" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name "nvflare.egg-info" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name "build" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name "dist" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name ".mypy_cache" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name ".pytype" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name ".coverage" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type f -name ".coverage.*" -exec rm -r "{}" \;
    find ${WORK_DIR} -depth -maxdepth 1 -type d -name "__pycache__" -exec rm -r "{}" \;
}

function torch_validate {
    echo "validate torch installation"
    python3 -c 'import torch; print(torch.__version__); print(torch.rand(5,3))'
}

function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}

function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
    echo "Please run auto style fixes: ${green}./runtests.sh -f {noColor}"
}
function report_status() {
    status="$1"
    if [ ${status} -ne 0 ]
    then
       print_style_fail_msg
       exit ${status}
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
  folders_to_check_license="nvflare tests"
  grep -r --include "*.py" --exclude-dir "*protos*" -L "\(# Copyright (c) \(2021\|2021-2022\|2022\), NVIDIA CORPORATION.  All rights reserved.\)\|\(This file is released into the public domain.\)" ${folders_to_check_license} > no_license.lst
  if [ -s no_license.lst ]; then
      # The file is not-empty.
      cat no_license.lst
      echo "License text not found on the above files."
      echo "Please fix them."
      rm -f no_license.lst
      exit 1
  else
      echo "All Python files in folder (${folders_to_check_license}) have license header"
      rm -f no_license.lst
  fi
}
function flake8_check() {
    #python3 -m flake8 nvflare
    echo "${separator}${blue}flake8${noColor}"
    python3 -m flake8 --version
    python3 -m flake8 "$1" --count --statistics
    report_status "$?"
}
function black_check() {
  echo "${separator}${blue}black-check${noColor}"
  python3 -m black --check "$1"
  report_status "$?"
  echo "Done with black code style checks"
}

function black_fix() {
  echo "${separator}${blue}black-fix${noColor}"
  python3 -m black "$1"
}

function isort_check() {
  echo "${separator}${blue}isort-check${noColor}"
  python3 -m isort --check "$1"
  report_status "$?"
}

function isort_fix() {
  echo "${separator}${blue}isort-fix${noColor}"
  python3 -m isort "$1"
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
        python3 -m pytype -j ${NUM_PARALLEL} --python-version="$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")" "$1"
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
  check_target="$1"
  # remove pylint for now
  # pylint_check  $check_target
  black_check   $check_target
  isort_check   $check_target
  flake8_check  $check_target
  # pytype causing check fails, comment for now
  # pytype_check  $check_target

  # gives a lot false alarm, comment out for now
  # mypy_check    $check_target
}

function fix_style_import() {
  fix_target="$1"
  black_fix "${fix_target}"
  isort_fix "${fix_target}"
}

################################################################################
export PYTHONPATH=${WORK_DIR}:$PYTHONPATH && echo "PYTHONPATH is ${PYTHONPATH}"

function help() {
    echo "Add description of the script functions here."
    echo
    echo "Syntax: runtests.sh  [-h|--help]
                               [-l|--check-license]
                               [-s|--check-format]
                               [-f|--fix-format]
                               [-u|--unit-tests]
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
    echo "    -r | --test-report            : used with -u command, turn on unit test report flag. It has no effect without -u "
    echo "    -c | --coverage               : used with -u command, turn on coverage flag,  It has no effect without -u "
    echo "    -d | --dry-run                : set dry run flag, print out command"
    echo "         --clean                  : clean py and other artifacts generated, clean flag to allow re-install dependencies"
#   echo "    -i | --integration-tests      : integration tests"
    exit 1
}

coverage_report=false
unit_test_report=false
dry_run_flag=false
flare_deps_installed=false
if [[ -f /tmp/.flare_deps_installed ]]; then
  flare_deps_installed=true
fi


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
            target="nvflare tests"
          fi
        ;;

      -f |--fix-format)
          if [ ! -z "${target}" ]; then
            cmd="fix_style_import ${target}"
          else
            cmd="fix_style_import nvflare && fix_style_import tests"
          fi
        ;;
      -c|--coverage)
          coverage_report=true
        ;;

      -r|--test-report)
          echo "set unit test flag"
          unit_test_report=true
        ;;

      -u |--unit*)
        cmd_prefix="torch_validate; python3 -m pytest --numprocesses=auto "

        echo "coverage_report=" ${coverage_report}
        if [ "${coverage_report}" == true ]; then
          cmd_prefix="${cmd_prefix} --cov=${target} --cov-report html:cov_html"
        fi

        if [ "${unit_test_report}" == true ]; then
          cmd_prefix="${cmd_prefix} --junitxml=unit_test.xml "
        fi
        cmd="$cmd_prefix"

        if [[ -z $target ]]; then
            target="tests/unit_test"
        fi

        ;;

       --clean)
         cmd="clean"
         ;;
       -d|--dry-run)
        dry_run_flag=true
        ;;

       -*)
          help
          exit
        ;;

    esac
    shift
done

if [[ -z $cmd ]]; then
   cmd="check_license -l;
        check_style_type_import nvflare tests;
        fix_style_import nvflare;
        fix_style_import tests ;
        python3 -m pytest --numprocesses=auto --cov=nvflare --cov-report html:cov_html --junitxml=unit_test.xml tests/unit_test;
       "
else
   cmd="$cmd $target"
fi

echo "running command: "
echo "                  install_deps "
echo "                 " "$cmd"
echo "                 "
if [[ $dry_run_flag = "true" ]]; then
    dry_run "$cmd"
else
    install_deps
    eval "$cmd"
fi
echo "Done"
