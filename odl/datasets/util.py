# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for datasets."""

import os
from os.path import join, expanduser, exists

try:
    # Python 2
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.request import urlopen

from shutil import copyfileobj
from scipy import io


__all__ = ('get_data_dir', 'get_data')


def get_data_dir():
    """Get the data directory."""
    base_odl_dir = os.environ.get('ODL_DIR', expanduser(join('~', 'odl')))
    data_home = join(base_odl_dir, 'datasets')
    if not exists(data_home):
        os.makedirs(data_home)
    return data_home


def get_data(filename, subset, url):
    """Get a dataset with from a url with local caching.

    Parameters
    ----------
    filename : str
        Name of the file, for caching.
    subset : str
        To what subset the file belongs (e.g. 'ray_transform'). Each subset
        is saved in a separate subfolder.
    url : str
        url to the dataset online.

    Returns
    -------
    dataset : dict
        Dictionary containing the dataset.
    """
    # check if this data set has been already downloaded
    data_dir = join(get_data_dir(), subset)
    if not exists(data_dir):
        os.makedirs(data_dir)

    filename = join(data_dir, filename)

    # if the file does not exist, download it
    if not exists(filename):
        print('data {}/{} not in local storage, downloading from {}'
              ''.format(subset, filename, url))

        # open the url of the data
        with urlopen(url) as data_url:
            # store downloaded file locally
            with open(filename, 'w+b') as storage_file:
                copyfileobj(data_url, storage_file)

    # load dataset file
    with open(filename, 'rb') as storage_file:
        data_dict = io.loadmat(storage_file)

    return data_dict

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
