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

Prometheus & Grafana
-------------------------
Launched prerequisites contain the Prometheus and Grafana monitoring toolkit.
While Prometheus is used for the application metrics scrapping, Grafana is used for the visualization purposes.
We prepared the default configuration files and a dashboard, which can be managed from the UI.
Those are added into containers via docker volumes mounting and you can make the changes in those folders locally
(and restart containers to apply changes).

``prerequisites/configs/prometheus.yml`` contains the Prometheus default configuration, you can check
`official documentation <https://prometheus.io/docs/prometheus/latest/configuration/configuration/>`_ for the in-depth
understanding and customization. However, you should not change the `master-agent` job targets to allow correct work of
Stalactite monitoring.

.. note::
    If you encounter the Grafana provisioning permissions problem in the Grafana container (on start), you should check
    the permissions set on the local configuration files.

``prerequisites/configs/grafana_provisioning`` contains default configuration files used by Grafana. **Datasources**
connect the Prometheus, **Dashboards** add default experimental dashboards from
``prerequisites/configs/grafana_provisioning/default_dashboards``. You can add your own files relying on the Grafana
documentation `for dashboards configuration <https://grafana.com/docs/grafana/latest/dashboards/>`_ and `datasources
management <https://grafana.com/docs/grafana/latest/datasources/>`_.

When you open the Grafana UI at ``graphana_host:grafana_port``, you will see the registration form, after you proceed,
at the ``Dashboards`` section, there is the default dashboard (`VFL experiment dashboard`) with default panels,
corresponding to metrics scraped from VFL master. Panel info contain description and some simple customizations, which
are available for it. You can also check the `Grafana panels documentation
<https://grafana.com/docs/grafana/latest/panels-visualizations/>`_ and create your own dashboards from scratch.

.. note::
    The default Grafana username and password: ``admin; admin``.


You can also customize the buckets for the Prometheus Histogram metrics at ``stalactite/communications/grpc_utils/utils.py``.
For example, if you want to have a more detailed info on the message size received by master:

.. code-block:: python

    class PrometheusMetric(enum.Enum):
        ...
        recv_message_size = Histogram(
            "task_message_size",
            "Size of the received by master messages (bytes)",
            ["experiment_label", "client_id", "task_type", "run_id"],
            buckets=np.concatenate([np.arange(0, 10000, 2500), np.array([10 ** pow for pow in range(4, 9)])])
        )
        ...

Change the values of the keyword argument buckets to more granular values (check the client
`docs <https://prometheus.github.io/client_python/instrumenting/histogram/>`_)
