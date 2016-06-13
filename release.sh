#!/bin/sh

function print_usage_and_quit {
    echo "Usage: $0 <version> [--dry-run]"
    exit 1
}

# Process command line args
if [ "$#" -eq 0 ]; then
    print_usage_and_quit
fi

set +x

PACKAGE_NAME="odl"

# Clumsy parsing of command line arguments
if [ "$#" == 1 ]; then
    if [ "$1" == "--dry-run" ]; then
        print_usage_and_quit
    fi
    DRY_RUN=0
    VERSION=$1
elif [ "$#" == 2 ]; then
    if [ "$1" == "--dry-run" ]; then
        VERSION=$2
    elif [ "$2" == "--dry-run" ]; then
        VERSION=$1
    else
        print_usage_and_quit
    fi

    DRY_RUN=1
else
    print_usage_and_quit
fi


### PyPI upload

PYPI_USER=odlgroup
TWINE=$(which twine)
if [ -z $TWINE ]; then
    echo 'Error: twine was not found. Please install it (e.g. via "pip install twine)".'
    exit 1
fi

PYPI_DIST_DIR=$(cd dist && echo "$PWD")  # Absolute path
PYPI_DIST_FILES="$(ls -1 $PYPI_DIST_DIR/${PACKAGE_NAME}-${VERSION}*)"

if [ $DRY_RUN -eq 1 ]; then
    echo "The following files would be uploaded to PyPI:"
    echo -e "$PYPI_DIST_FILES"
else
    $TWINE upload -u $PYPI_USER $PYPI_DIST_FILES || exit 1
fi

echo

### Conda upload
CONDA_USER="odlgroup"

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
    echo "The following files would be uploaded to Anaconda Cloud:"
    echo -e $CONDA_DIST_FILES
else
    anaconda login --username $CONDA_USER || exit 1
    anaconda upload $CONDA_DIST_FILES || exit 1
    anaconda logout
fi
