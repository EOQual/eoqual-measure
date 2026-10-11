#!/bin/bash

# Get root directory :
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Activate env
source ${ROOTDIR}/__env__/bin/activate

# SRC in pythonpath
export PYTHONPATH=${PYTHONPATH}:${ROOTDIR}/../src
