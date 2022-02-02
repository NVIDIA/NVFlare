#!/usr/bin/env bash

export projectname='nvflare_brats18'
export projectpath="."

# python3 -m venv ${projectname}
virtualenv -p python3.8 $projectname
source ${projectname}/bin/activate
