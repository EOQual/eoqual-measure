#!/bin/bash

WANTED_PYTHON_VERSION="3.12"

# Test if python is available in the correct version :
command -v python >/dev/null 2>&1
if [[ $? == 1 ]]
then
    echo "python ${WANTED_PYTHON_VERSION} must be available."
    exit 1
fi

PYTHON_VERSION=`python -V | cut -d" " -f2 | cut -d"." -f1-2`
if [[ ${PYTHON_VERSION} != ${WANTED_PYTHON_VERSION} ]]
then
    echo "python ${WANTED_PYTHON_VERSION} must be available."
    exit 1
fi


# Get root directory :
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Create venv in root directory :
cd ${ROOTDIR}
echo "Creating venv..."
python -m venv __env__
. ./__env__/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip 1>/dev/null 2>&1

# Installing Poetry
echo "Installing poetry..."
pip install poetry 1>/dev/null 2>&1

# Dependencies :
# echo "Installing dependencies..."
poetry install --no-root --with=docs --with=dev -v

