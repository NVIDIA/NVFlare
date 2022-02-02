#!/usr/bin/env bash

export projectname='nvflare_prostate'
export projectpath="."

# python3 -m venv ${projectname}
virtualenv -p python3.8 $projectname
source ${projectname}/bin/activate
