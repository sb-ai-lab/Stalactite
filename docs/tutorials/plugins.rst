.. _plugins_tutorial:

*how-to:* Implement your own ML-algorithm (plugin)
====================================================

In the :ref:`master_types`, the implemented algorithms are listed. If you want to incorporate your own
logic into the framework, you should write the agents classes furnished with the specifications on your algorithm.
For the framework to find the plugins you write, you should create (or use existing) folder `plugins`
alongside the sourcecode of the `Stalactite`.



.. code-block:: bash

    |-- ...
    |-- plugins
    |-- stalactite
    `-- ...

In the `plugins` folder, create a folder containing your agents. The name of this folder does not matter, but
it is important for the agent implementation discovery to name your files correctly:

- the master class implementation should be placed in a file named: `party_master.py`
- the member class implementation should be placed in a file named: `party_member.py`
- the arbiter (if implemented) should be placed in a file named: `party_arbiter.py`

We have copied the honest logistic regression implementation into the repository `plugins` folder for you to see as the example.

At runtime, to use the plugin in the experiment, the configuration file must be adjusted accordingly. For example, to make the framework use
the honest logistic regression implementation from the plugins folder, you should change the ``vfl_model.vfl_model_name``
to the path from `plugins` to your directory with agents' files.


.. code-block:: yaml

    vfl_model:
      vfl_model_name: plugins.logistic_regression

After performing the aforementioned steps, the framework should be able to discover implemented agents and will use them
in the experiment.