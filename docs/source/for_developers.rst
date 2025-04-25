For Developers
==============================================================================

This section elaborates on the design of THUNER, describing some key internal classes
and functions. 

Input Records
-------------------------------------------------

A core challenge I found when working with MINT, and other tracking algorithms, was
the need to iterate over multiple lists of files corresponding to distinct datasets, 
Note that for tracking algorithms, we really need to iterate over time-steps, but 
sometimes a single file will contain multiple timesteps; the following classes address 
this.

.. autopydantic_model:: thuner.track._utils.BaseInputRecord

.. autopydantic_model:: thuner.track._utils.TrackInputRecord

.. autopydantic_model:: thuner.track._utils.InputRecords

Tracks
-------------------------------------------------------------------------------

We also need classes that collect attributes and other data for each object as the 
tracking run proceeds. First we need a class to store the attributes of each object
as they are collected. Attributes are stored in dictionaries, with the dictionaries
cleared periodically when data is written to disk.
:class:`thuner.option.track.TrackOptions`.

.. autopydantic_model:: thuner.attribute.utils.AttributesRecord
    :no-index:

Now we need classes to manage each object and level, noting these classes are nested in
a manner analogous to the :class:`thuner.option.track.ObjectOptions`, 
:class:`thuner.option.track.LevelOptions` and :class:`thuner.option.track.TrackOptions` 
classes. Note we also store the corresponding options in each tracking class. 

.. autopydantic_model:: thuner.track._utils.ObjectTracks

.. autopydantic_model:: thuner.track._utils.LevelTracks

.. autopydantic_model:: thuner.track._utils.Tracks