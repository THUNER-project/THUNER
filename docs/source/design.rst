Design
================

Introduction
----------------
THUNER began as a `fork of the TINT repository <https://github.com/THUNER-project/TINT>`__, 
and was originally called `Mesoscale TINT (MINT) <https://doi.org/10.1175/MWR-D-22-0146.1>`__. 
The idea behind MINT was to identify multiple objects associated with a single storm, 
and group these together into new storm "system" objects, tracking these systems in time. 
These storm systems were also "tagged" with attributes from other datasets,
e.g. ambient wind and temperature fields from the European Centre for Medium-Range
Weather Forecasts (ECMWF) reanalysis. 

As work on MINT progressed, the codebase became increasingly cumbersome, and it
became necessary to reimplement the core features in other ways. While reimplementation
into other existing packages was considered, in the end it seemed simpler to just
re-write the MINT codebase from scratch as a new project, THUNER. 

Objectives
----------------
The core goals of THUNER include the following.

#. **To be input agnostic.** While storm tracking algorithms were originally developed
   for radar data, today one needs to apply such algorithms to many distinct datasets, and 
   even use different datasets for different attributes of the same object, within the 
   same workflow.
#. **To support distinct methodologies and be easily extensible.** Object based 
   analysis of meteorological phenomena is inherently ambiguous. As noted by 
   `Dawe and Austin (2012) <https://doi.org/10.5194/acp-12-1101-2012>`__, 

      A cloud is a process, not an object; a rising parcel of moist air may
      condense, a parcel of air containing condensate may evaporate, and a
      cloud may merge with another cloud or split into multiple clouds.

   Given this ambiguity, a `very large variety <https://doi.org/10.1029/2023JD040254>`__ 
   of object based analysis methodologies now exist in the published literature. 
   Ideally, a scientist should test sensitivity of conclusions to ambiguous methodological 
   choices. THUNER therefore aims to organize the many distinct detection, grouping, tagging 
   and analysis methodologies, and allow methodological choices to be set and 
   recorded before each tracking run, the intent being to simplify sensitivity testing
   and ensure reproducibility.
#. **To support multiple coordinate systems.** While individual radar domains can be
   analysed in cartesian coordinates, larger scale datasets require geographic coordinates.
#. **To support parallelisation.** The tracking process can be greatly accelerated using 
   parallelisation, particularly for large datasets.


.. figure:: ./images/packages.png
   :alt: Packages

   Subpackages and modules and their import relationships.