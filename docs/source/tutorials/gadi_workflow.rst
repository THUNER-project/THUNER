Working Remotely: NCI Gadi
--------------------------------------

Australian users can run THUNER on the `National Computational Infrastructure (NCI) <https://nci.org.au/>`__
high performance computing system, `Gadi <https://nci.org.au/our-systems/hpc-systems>`__ .
Scripts for working with THUNER on Gadi are available in the 
`workflow/gridrad_severe_gadi <https://github.com/THUNER-project/THUNER/tree/main/workflow/gridrad_severe_gadi>`__
folder on the `THUNER GitHub repository <https://github.com/THUNER-project/THUNER>`__.

The simplest way to install THUNER on Gadi is to first install a standalone version of
`conda <https://www.anaconda.com/docs/getting-started/miniconda/install>`__ in your
``g/data/<project>/<user>`` directory. Ask the computational support team if you need help
doing this. You can then clone the THUNER repository to your ``g/data/<project>/<user>``
directory, and create the THUNER ``conda`` environment, as described in the :ref:`from-github`
section of the :doc:`Installation <../installation>` page. After cloning you can modify
the scripts in the the `workflow/gridrad_severe_gadi <https://github.com/THUNER-project/THUNER/tree/main/workflow/gridrad_severe_gadi>`__
folder so that paths etc are correct. 

Note the GridRad Severe data has been copied to Gadi tape storage under the v46 project, 
and currently exists in the directory ``esh563/d841006/volumes/<year>``, with each GridRad 
Severe "event" a compressed TAR file. Years can be copied to ``scratch`` and extracted
by navigating on the command-line to the ``workflow/gridrad_severe_gadi`` directory in 
the cloned THUNER repository, and running the command

.. code-block:: shell

    sh mdss_year_get.sh <year>

replacing ``<year>`` with the actual year you need, e.g. 2010. 

THUNER can be run with the same options as the GridRad Severe 
`demo notebook <https://github.com/THUNER-project/THUNER/blob/main/demo/gridrad.ipynb>`_
using the script

.. code-block:: shell

    sh gridrad_year_job.sh <year>

Be wary that processing all 13 years will consume 20-30 KSU. Note that all 13 years have 
already been processed with these options, and are available on tape under the v46 group
at ``esh563/gridrad_severe.tar.gz``. You can set different options by modifying the ``gridrad.py`` file in the 
``workflow/gridrad_severe_gadi`` directory. In particular, commenting out line 102, 
``visualize.attribute.mcs_series(*args, **kwargs)``, will switch off the generation of 
figures, and will reduce the KSU cost by at least half.

If required, you can then run the ``copy_local.sh`` script on your local machine to 
copy data to local disk. The ``workflow/gridrad_severe_analysis`` contains scripts
for analyzing the data in a fashion analogous to 
Ewan Short's `previously published work <https://orcid.org/0000-0003-2821-8151>`_