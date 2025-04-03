Design
================

Introduction
----------------
The Thunderstorm Event Reconnaissance (THUNER) package is an evolution of the 
`TINT is not TITAN (TINT) <https://doi.org/10.1175/JAMC-D-20-0119.1>`__ codebase, with 
TINT itself an evolution of the `Thunderstorm Identification, Tracking, Analysis and 
Nowcasting (TITAN) <https://ral.ucar.edu/solutions/products/thunderstorm-identification-tracking-analysis-and-nowcasting-titan>`__ codebase.
THUNER began as a `fork of the TINT repository <https://github.com/THUNER-project/TINT>`__, 
and was originally called `Mesoscale TINT (MINT) <https://doi.org/10.1175/MWR-D-22-0146.1>`__. 
The idea behind MINT was to identify multiple objects associated with a single storm, 
and group these together into new storm "system" objects, tracking these systems in time. 
These storm systems were also "tagged" with attributes from other datasets,
e.g. ambient wind and temperature fields from the European Centre for Medium-Range
Weather Forecasts (ECMWF) reanalysis. 

As work on MINT progressed, the codebase became increasingly cumbersome, and I
concluded I would eventually need to reimplement the core features in other ways. While
I considered reimplementation into another existing package, e.g. tobac or PyFLEXTRKR, 
but eventually decided it would be better to simply re-write my existing MINT code 
from scratch. 


.. figure:: ./images/packages.png
   :alt: Packages

   Subpackages and modules in ``thuner``, and their import relationships.