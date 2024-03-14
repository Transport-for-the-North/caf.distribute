.. caf.toolkit documentation master file, created by
   sphinx-quickstart on Wed Oct  4 13:40:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to caf.distribute's documentation!
=======================================

A module of CAF for distributing demand across matrices.

The Common Analytical Framework (CAF) is a collection of transport planning and appraisal functionalities. It's part of a project to make useful transport related functionality more widely available and easily accessible. Other modules include:

* `CAF.Toolkit <https://caftoolkit.readthedocs.io/en/latest/>`_

* `CAF.Space <https://cafspace.readthedocs.io/en/latest/>`_

Tool info
---------

CAF.distribute focusses on tools and models to distribute vectors of data into matrices.
Currently, it only contains a self calibrating gravity model that is capable of calibrating data to a single cost distribution area.

Future plans involve enhancing this gravity model to handle multiple cost distribution areas, that is if the gravity model data covered two counties, a different cost distribution could be used for each county, while ensuring all totals still match.
There are also plans to move the `Iterative Proportional Fitting <https://en.wikipedia.org/wiki/Iterative_proportional_fitting>`_ algorithm over that currently exists in `CAF.Toolkit <https://github.com/Transport-for-the-North/caf.toolkit/blob/3c7ba4b7d770e90fcadb81672c9c485bcf08cecf/src/caf/toolkit/iterative_proportional_fitting.py#L837>`_.

Installation
------------
caf.toolkit can be installed either from pip or conda forge:

``pip install caf.distribute``

``conda install caf.distribute -c conda-forge``

.. toctree::
   :maxdepth: 4
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Modules
-------

.. toctree::
   :maxdepth: 1

   array_utils
   cost_functions
   furness
   iterative_proportional_fitting
   utils

Sub-Packages
------------

.. toctree::
   :maxdepth: 1

   gravity_model
