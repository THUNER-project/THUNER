Usage
=====

.. _installation:

Installation
------------

To use THUNER, first install it using pip

.. code-block:: console

   (THUNER) $ pip install thuner

or conda

.. code-block:: console

   (THUNER) $ conda install thuner

Tracking
--------

The core tracking function is ``thuner.track.track.track()``:

.. autofunction:: thuner.track.track.track
   :noindex:

This function takes a series of options objects as input.

For example:

>>> import thuner
>>> thuner.track.track.track()

