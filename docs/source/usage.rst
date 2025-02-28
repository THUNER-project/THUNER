Usage
=====

Installation
------------

The `THUNER repository <https://github.com/THUNER-project/THUNER>`__ can be cloned from 
GitHub in the usual ways. Cloning the repository is the easiest way to access the demo, 
workflow and gallery folders. After cloning, a new conda environment using 
`environment.yml`, then run `pip install .` from the THUNER root directory.

Alternatively, THUNER can be installed using `conda`, ideally into a new environment:

.. code-block:: console

   (THUNER) $ conda install -c conda-forge thuner

While `conda` installation is preferred, `pip` may also be used. First install the `esmpy` 
package manually as detailed
`here <https://xesmf.readthedocs.io/en/latest/installation.html#notes-about-esmpy>`__.
THUNER can then be installed using

.. code-block:: console

   (THUNER) $ pip install thuner

Note that THUNER depends on `xesmf`` for regridding, and is therefore currently only
available on Linux and OSX systems.

Design Philosophy
------------------

THUNER aspires to provide a stable foundation for exploring complex and diverse 
tracking and analysis ideas. 

Tracking
--------

The core tracking function is ``thuner.track.track.track()``:

.. autofunction:: thuner.track.track.track
   :noindex:

This function takes a series of options objects as input.

For example:

>>> import thuner
>>> thuner.track.track.track()

