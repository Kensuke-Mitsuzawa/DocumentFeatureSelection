Installing / Upgrading
======================

Installing from source
----------------------

If you prefer install directly from the source::

  $ cd document-feature-selection
  $ sudo python setup.py install

Creating packages
-----------------

You can easily create documentation and packages::

  $ cd document-feature-selection
  $ python setup.py sdist  # generate source .tar.gz file
  $ python setup.py bdist_deb  # require python-all and python-stdeb packages
  $ python setup.py bdist_rpm  #
  $ python setup.py bdist_msi  # generate a Windows installer
  $ python setup.py bdist  # generate a binary .tar.gz
  $ python setup.py py2exe  # generate a portable Windows application
  $ python setup.py py2app  # generate a portable Mac OS X application
