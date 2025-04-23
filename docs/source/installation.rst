Installation
-----------------
THUNER uses `conda <https://www.anaconda.com/docs/getting-started/miniconda/install>`__ 
or `pip <https://pypi.org/project/pip/>`__ for installation and to manage dependencies. 
First ensure either ``conda`` or ``pip`` is installed; ``conda`` is the preferred method. Note 
that THUNER depends on ``xesmf`` for regridding, which is not currently supported on 
Windows. While THUNER can still be installed on Windows systems, some features may not
work as intended.


.. _from-github:

From GitHub
~~~~~~~~~~~~

The `THUNER repository <https://github.com/THUNER-project/THUNER>`__ can be cloned from 
GitHub in the usual ways. Cloning the repository is the easiest way to access the demo, 
workflow and gallery folders. After cloning, navigate to the THUNER directory and create
a new conda environment using 

.. code-block:: console

   conda env create -f environment.yml
   conda activate thuner

Then run 

.. code-block:: console

   pip install . 

from the THUNER root directory.

From conda-forge
~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, THUNER can be installed using ``conda``. Create a new conda environment
as above, then run

.. code-block:: console

   conda install -c conda-forge thuner

From PyPI
~~~~~~~~~~~~~~~~~~~~~~~
While ``conda`` installation is preferred, ``pip`` may also be used to install directly
from PyPI. First install the ``esmpy`` package manually as detailed
`here <https://xesmf.readthedocs.io/en/latest/installation.html#notes-about-esmpy>`__.
Then run

.. code-block:: console

   pip install thuner