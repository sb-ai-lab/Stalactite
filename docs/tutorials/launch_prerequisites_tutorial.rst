.. _prerequisites_tutorial:

*how-to:* Manage prerequisites
====================================================

After you defined the configuration file (including ``prerequisites`` and ``docker`` sections), you are able to launch
the MlFlow and Prometheus containers.

.. warning::
    Host of prerequisites' containers must be the same to the master host of the distributed experiment, otherwise, the
    Prometheus won't be able to scrape metrics from the master.

Now you can launch the containers by running (the `-d` or `--detached` option launches the containers in the background)

.. code-block:: bash

    stalactite prerequisites start --config-path <path-to-defined/config.yml> [-d]

To stop the prerequisites run:

.. code-block:: bash

    stalactite prerequisites stop --config-path <path-to-defined/config.yml> [--remove] [--remove-volumes]
