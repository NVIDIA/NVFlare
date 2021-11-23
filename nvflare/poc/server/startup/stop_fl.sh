#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Shutdown request created.  FL system will shutdown soon."
touch $DIR/../shutdown.fl
