Thunderstorm Event Reconnaissance (THUNER)
==========================================

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/mcs_gridrad_20100804.gif
   :alt: GridRad Demo

Package description
-------------------

Welcome to the Thunderstorm Event Reconnaissance (THUNER) package!
THUNER is a flexible toolkit for multi-feature detection, tracking,
tagging and analysis of events in meteorological datasets; documentation is 
`available online <https://thuner.readthedocs.io/en/latest/>`__.
THUNER's intended application is the tracking and analysis convective weather events. 
If you use THUNER in your work, consider citing 

- `Leese et al. (1971) <https://doi.org/10.1175/1520-0450\(1971\)010\<0118:AATFOC\>2.0.CO;2>`__
- `Dixon and Wiener (1993) <https://doi.org/10.1175/1520-0426\(1993\)010\<0785:TTITAA\>2.0.CO;2>`__
- `Whitehall et al. (2015) <https://doi.org/10.1007/s12145-014-0181-3>`__
- `Fridlind et al (2019) <https://doi.org/10.5194/amt-12-2979-2019>`__
- `Raut et al (2021) <https://doi.org/10.1175/JAMC-D-20-0119.1>`__
- `Short et al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__

Note many excellent alternatives to THUNER exist, including 
`PyFLEXTRKR <https://github.com/FlexTRKR/PyFLEXTRKR>`__, 
`GTG <https://github.com/kwhitehall/grab-tag-graph>`__,
`TAMS <https://github.com/knubez/TAMS>`__,
`tobac <https://github.com/tobac-project/tobac>`__ and 
`MOAAP <https://github.com/AndreasPrein/MOAAP>`__. When designing a tracking based 
research project involving THUNER, consider performing sensitivity tests using these 
alternatives.

Installation
------------

The `THUNER repository <https://github.com/THUNER-project/THUNER>`__ can be cloned from 
GitHub in the usual ways. Cloning the repository is the easiest way to access the demo, 
workflow and gallery folders. After cloning, a new conda environment using 
`environment.yml`, then run `pip install .` from the THUNER root directory.

Alternatively, THUNER can be installed using `conda`, ideally into a new environment:

.. code:: sh

   conda install -c conda-forge thuner

While `conda` installation is preferred, `pip` may also be used. First install the `esmpy` 
package manually as detailed
`here <https://xesmf.readthedocs.io/en/latest/installation.html#notes-about-esmpy>`__.
THUNER can then be installed using

.. code:: sh

   pip install thuner

Note that THUNER depends on `xesmf` for regridding, and is therefore currently only
available on Linux and OSX systems.

Examples
--------

GridRad
~~~~~~~

The examples below illustrate the tracking of convective systems in
`GridRad Severe <https://gridrad.org/>`__ radar data. Object merge
events are visualized through the “mixing” of the colours associated
with each merging object. Objects that split off from existing objects
retain the colour of their parent object. Objects which intersect the
domain boundary have their
`stratiform-offsets <https://doi.org/10.1175/MWR-D-22-0146.1>`__ and
velocities masked, as these cannot be measured accurately when the
object is partially outside the domain.

The example below depicts multiple
`trailing-stratiform <https://doi.org/10.1175/1520-0493(2001)129%3C3413:OMOMMC%3E2.0.CO;2>`__
type systems.

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/mcs_gridrad_20100804.gif
   :alt: GridRad Demo


The example below depicts multiple
`leading-stratiform <https://doi.org/10.1175/1520-0493(2001)129%3C3413:OMOMMC%3E2.0.CO;2>`__
type systems.

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/mcs_gridrad_20100120.gif
   :alt: GridRad Demo


Etymology
---------

According to `Wikipedia <https://en.wikipedia.org/wiki/Thor>`__, between
the 8th and 16th centuries the storm god more commonly known as Thor was
called “Thuner” by the inhabitants of what is now west Germany.


Acknowledgements
------------------

THUNER's documentation is hosted on Read the Docs. THUNER was developed by 
`Ewan Short <https://orcid.org/0000-0003-2821-8151>`__ while supported by 
Australian Research Council grants 
`CE170100023 <https://dataportal.arc.gov.au/NCGP/Web/Grant/Grant/CE170100023>`__.
and `DP200102516 <https://dataportal.arc.gov.au/NCGP/Web/Grant/Grant/DP200102516>`__. 
Computational resources during development were provided by the Australian
`National Computational Infrastructure (NCI) <https://nci.org.au/>`__.
