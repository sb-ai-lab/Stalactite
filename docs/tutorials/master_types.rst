.. _master_types:

*info:* Implemented algorithms' master types
====================================================

For correct data preparation and inference launch, one should understand the experiment master behaviour. While members
always have a trainable model with weights and part of the data (features), master can either hold a meta-model with
non-changing on the training weights or be an experimental labels holder only.

The following table demonstrates the implemented algorithms master structure.

- The column "Model name" describes the implemented model type.
- "ML-type" holds the agents world structure info.
- "Trainable model" identifies the type of the master for the inference (`False` is for the ``master_no_model``, `True` can be interpreted as the ``master_with_model``), namely you don't need to pass the ``model_path`` for the ``master_no_model`` type, because master does not hold the trainable weights, and vice versa.
- "Features" column shows whether on the data preparation, master preprocessor must contain any features from dataset.


+--------------+------------+-----------------+-----------+
|  Model name  |  ML-type   | Trainable model | Features  |
+==============+============+=================+===========+
|    linreg    | honest     | False           | False     |
+--------------+------------+-----------------+-----------+
|              | honest     | False           | False     |
+    logreg    +------------+-----------------+-----------+
|              | arbitered  | True            | True      |
+--------------+------------+-----------------+-----------+
|     mlp      | honest     | True            | False     |
+--------------+------------+-----------------+-----------+
|    resnet    | honest     | True            | False     |
+--------------+------------+-----------------+-----------+
| efficientnet | honest     | True            | False     |
+--------------+------------+-----------------+-----------+

You might want to check out the following tutorials:

- :ref:`inference_tutorial`
- :ref:`data_proc_tutorial`
