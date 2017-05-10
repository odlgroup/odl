# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for ODL datasets."""

import os
from os.path import join, expanduser, exists

try:
    # Python 2
    from urllib2 import HTTPError
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.error import HTTPError
    from urllib.request import urlopen
    
from shutil import copyfileobj
from scipy import io

def get_data_dir():
    """Get the data directory."""
    data_home = expanduser(join('~', 'odl_data'))
    if not exists(data_home):
        os.makedirs(data_home)
    return data_home


def get_data(filename, subset, url):
    """Download a dataset."""

    # check if this data set has been already downloaded
    data_dir = get_data_dir()
    data_dir = join(data_dir, subset)
    if not exists(data_dir):
        os.makedirs(data_dir)

    filename = join(data_dir, filename)

    # if the file does not exist, download it
    if not exists(filename):
        print('data {}/{} missing, downloading from {}'
              ''.format(subset, filename, url))
        try:
            data_url = urlopen(url)
        except HTTPError as e:
            if e.code == 404:
                e.msg = "Dataset '%s' not found on mldata.org." % dataname
            raise
        # store downloaded file
        try:
            with open(filename, 'w+b') as storage_file:
                copyfileobj(data_url, storage_file)
        except Exception as e:
            print(e)
            os.remove(filename)
            raise
        data_url.close()

    # load dataset matlab file
    with open(filename, 'rb') as storage_file:
        data_dict = io.loadmat(storage_file)

    return data_dict