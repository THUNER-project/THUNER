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

.. autopydantic_model:: thuner.attribute.utils.AttributesRecord
    :no-index:

We also need classes to manage each object and level, noting these classes are nested in
a manner analogous to the :class:`thuner.option.track.ObjectOptions`, 
:class:`thuner.option.track.LevelOptions` and :class:`thuner.option.track.TrackOptions` 
classes. Note we also store the corresponding options in each tracking class. 

.. autopydantic_model:: thuner.track._utils.ObjectTracks

.. autopydantic_model:: thuner.track._utils.LevelTracks

.. autopydantic_model:: thuner.track._utils.Tracks

Visualization
-------------------------------------------------------------------------------

Another challenge with MINT and other packages was visualization. I like generating lots of figures/animations to check my tracking/analysis algorithms are doing what I expect (more or less). Such checks require object masks, domain boundaries, and object attributes like flow velocity and major axes be visualized. I also like to create panelled figures showing the different objects comprising grouped objects (e.g. convective and stratiform echoes). This can lead to very complex and confusing plotting functions.

THUNER aims to organize visualization code in a more modular way. Figures are managed internally by be classes like :class:`thuner.visualize.attribute.BaseFigure` and :class:`thuner.visualize.attribute.GroupedObjectFigure`. 

.. autopydantic_model:: thuner.visualize.attribute.BaseFigure

.. autopydantic_model:: thuner.visualize.attribute.GroupedObjectFigure

Figure layout is managed separately by classes like the following.

.. autoclass:: thuner.visualize.utils.BaseLayout

.. autoclass:: thuner.visualize.utils.Panelled

.. autoclass:: thuner.visualization.horizontal.PanelledUniformMaps

Attribute visualization is also separated out from the core figure generation code. Each attribute is managed by an instance of :class:`thuner.utils.AttributeHandler`. The required attribute handlers are then provided as figure options, allowing them to be swapped in and out as required.

.. autopydantic_model:: thuner.utils.AttributeHandler

Some typical combinations of figure options and attribute handlers are included in the :mod:`thuner.default` module.

