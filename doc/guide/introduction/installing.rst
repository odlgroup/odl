##########
Installing
##########

Installing odl is intended to be straightforward, and this guide is meant for new users. To install odl you need to do the following steps:

1. `Install Python`_
2. `Install Git`_
3. `Install ODL`_
4. `Run tests`_

If you have done any step before, you can ofc skip it.

Install Python
==============
To begin with, you need a python distribution. If you are an experienced user, you can use any distribution you'd like. If you are a python novice, we recommend that you install a full package such as Anaconda. 

Anaconda
--------
To install Anaconda

1. Download Anaconda from `anaconda's webpage <https://www.continuum.io/downloads>`_
2. Once installed run in a console
		``user$ conda update --all``
	to make sure you have the latest versions of all packages
	
Install Git
===========

You also need to install Git to be able to download `odl`

Overview
--------

================ =============
Debian / Ubuntu  ``sudo apt-get install git``
Fedora           ``sudo yum install git``
Windows          Download and install msysGit_
OS X             Use the git-osx-installer_
================ =============

.. _msysgit: http://code.google.com/p/msysgit/downloads/list
.. _git-osx-installer: http://code.google.com/p/git-osx-installer/downloads/list

In detail
---------

See the git page for the most recent information.

Have a look at the github install help pages available from `github help`_

There are good instructions here: http://book.git-scm.com/2_installing_git.html

.. _github help : https://help.github.com/

Install ODL
===========

You are now ready to install ODL! To do that, run the following where you want to install it

	``git clone https://github.com/odlgroup/odl``
	
	``cd odl``
	

For installation in a local user folder run

	``user$ pip install --user .``

For system-wide installation, run (as root, e.g. using `sudo` or equivalent)

	``root# pip install .``

(Optional) Install ODLpp
========================

If you also wish to use the (optional) CUDA extensions you need to run

	``user$ git submodule update --init --recursive``
	
	``user$ cd odlpp``

From here follow the instructions in odlpp and install it. You then need to re-install ODL.

Run tests
=========

To verify your installation you should run some basic tests. To run these:

	``user$ python run_tests.py``
