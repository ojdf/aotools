AOtools
=======

Useful tools for Adaptive Optics analysis for the Python Programming Language. If using this code for a publication please cite the `aotools paper <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-22-31316>`_ (M. J. Townson, O. J. D. Farley, G. Orban de Xivry, J. Osborn, and A. P. Reeves, "AOtools: a Python package for adaptive optics modelling and analysis," Opt. Express 27, 31316-31329 (2019))

.. image:: https://anaconda.org/aotools/aotools/badges/installer/conda.svg
   :target: https://conda.anaconda.org/aotools

.. image:: https://github.com/AOtools/aotools/actions/workflows/unit_test.yml/badge.svg
   :target: https://github.com/AOtools/aotools/actions/workflows/unit_test.yml

.. image:: https://ci.appveyor.com/api/projects/status/hru9gl4jekcwtm6l/branch/master?svg=true
   :target: https://ci.appveyor.com/project/Soapy/aotools/branch/master

.. image:: https://codecov.io/gh/AOtools/aotools/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/AOtools/aotools
  
.. image:: https://readthedocs.org/projects/aotools/badge/?version=v1.0.1
   :target: https://aotools.readthedocs.io/en/v1.0.1/?badge=v1.0.1
   :alt: Documentation Status

Required libraries
------------------

.. code-block:: python

   python
   SciPy
   NumPy
   matplotlib
   numba

Installation
------------

As everything is just pure python, you don't really need to "install" at all. To be able to use the tools from anywhere on your system,
add the ``aotools`` directory to your ``PYTHONTPATH``.
Alternatively you can use one of the methods below.

Anaconda
++++++++

AOtools can be installed in an anaconda environment using:

.. code-block:: python

   conda install -c aotools aotools

Pip
+++

AOtools can be installed using pip:

.. code-block:: python

    pip install aotools

(which may require admin or root privileges)

From Source
+++++++++++
Alternatively, to install the tools to your system python distribution from source, run:

.. code-block:: python

    python setup.py install

(which may require admin or root privileges) from the ``aotools`` directory.

Documentation
+++++++++++++
Full documentation is hosted by  `Read the Docs <https://aotools.readthedocs.io/en/v1.0.1/>`_

Usage Stats
-----------
Pip
+++
.. image:: https://img.shields.io/badge/dynamic/json.svg?color=bright%20green&label=Downloads%2FMonth&query=%24.data.last_month&url=https%3A%2F%2Fpypistats.org%2Fapi%2Fpackages%2Faotools%2Frecent
   :target: https://pypistats.org/packages/aotools
   
Anaconda
++++++++
.. image:: https://anaconda.org/aotools/aotools/badges/downloads.svg
   :target: https://anaconda.org/aotools/aotools
