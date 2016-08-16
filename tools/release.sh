#!/bin/bash

print_usage_and_quit() {
    echo "Usage: $0 <version> [--dry-run]"
    exit 1
}

if [ ! -f "setup.py" ]; then
    echo "This script must be run in the directory where the setup.py file lies."
    exit 1
fi

# Process command line args
if [ "$#" -eq 0 ]; then
    print_usage_and_quit
fi

# This is for debugging ("set -x" enables verbose printing)
set +x

PACKAGE_NAME="odl"

# Clumsy parsing of command line arguments
if [ "$#" = 1 ]; then
    if [ "$1" = "--dry-run" ]; then
        print_usage_and_quit
    fi
    DRY_RUN=0
    VERSION=$1
elif [ "$#" = 2 ]; then
    if [ "$1" = "--dry-run" ]; then
        VERSION=$2
    elif [ "$2" = "--dry-run" ]; then
        VERSION=$1
    else
        print_usage_and_quit
    fi

    DRY_RUN=1
else
    print_usage_and_quit
fi


if [ ! -d "dist" ]; then
    echo 'The `dist` directory does not exist.'
    echo 'You need to run `python setup.py sdist` and `python setup.py bdist_wheel` to build the distribution.'
    exit 1
fi


### PyPI upload

PYPI_DIST_DIR=$(cd dist && echo "$PWD")  # Absolute path
PYPI_DIST_FILES="$(ls -1ad $PYPI_DIST_DIR/* | grep ${PACKAGE_NAME}-${VERSION})"

if [[ -z $PYPI_DIST_FILES ]]; then
    echo 'No distribution files found in the `dist` directory.'
    echo 'You need to run `python setup.py sdist` and `python setup.py bdist_wheel` first to build the distribution.'
    exit 1
fi

if [ $DRY_RUN -eq 1 ]; then
    echo ""
    echo "The following files would be uploaded to PyPI:"
    echo -e "$PYPI_DIST_FILES"
else
    PYPI_USER=odlgroup
    TWINE=$(which twine)
    if [ -z $TWINE ]; then
        echo 'Error: twine was not found. Please install it (e.g. via `pip install twine`).'
        exit 1
    else
        $TWINE upload -u $PYPI_USER $PYPI_DIST_FILES || exit 1
    fi
fi


### Conda upload
CONDA_USER="odlgroup"
ANACONDA=$(which anaconda)

# Find conda build directory
if [ -n $CONDA_ENV_PATH ]; then
    CONDA_BUILD_DIR="$CONDA_ENV_PATH/../../conda-bld"
else
    CONDA_BUILD_DIR="$(which conda)/../../conda-bld"
fi

# Compile all files in the build folders corresponding to package and version
CONDA_DIST_FILES=""
CONDA_DIST_DIRS="linux-32 linux-64 osx-64 win-32 win-64"
for DIR in $CONDA_DIST_DIRS; do
    if [ -d $CONDA_BUILD_DIR/$DIR ]; then
        if [ -z $CONDA_DIST_FILES ]; then
            CONDA_DIST_FILES="$(ls -1 $CONDA_BUILD_DIR/$DIR/$PACKAGE_NAME-$VERSION*)\n"
        else
            CONDA_DIST_FILES="$CONDA_DIST_FILES$(ls -1 $CONDA_BUILD_DIR/$DIR/$PACKAGE_NAME-$VERSION*)\n"
        fi
    fi
done

# Do the upload (or pretend to)
if [ $DRY_RUN -eq 1 ]; then
    echo ""
    echo "The following files would be uploaded to Anaconda Cloud:"
    echo -e $CONDA_DIST_FILES
else
    if [ -z $ANACONDA ]; then
        echo 'anaconda was not found. Skipping upload to Anaconda Cloud.'
        exit 0
    else
        if [ -z $CONDA_DIST_FILES ]; then
            echo "Nothing to upload. Exiting."
            exit 0
        fi
        $ANACONDA login --username $CONDA_USER || exit 1
        $ANACONDA upload $CONDA_DIST_FILES || exit 1
        $ANACONDA logout
    fi
fi
