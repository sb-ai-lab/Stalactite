.. _data_proc_tutorial:

*how-to:* Preprocess data
======================================

When experiment runs from "local_experiment.py" file, you can use "load_processors" function:


.. autofunction:: examples.utils.local_experiment.load_processors

If the dataset is MNIST, then use "ImagePreprocessor":

stalactite.data_preprocessors.image_preprocessor
------------------------------------------------------

.. automodule:: stalactite.data_preprocessors.image_preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

The data supplied to the ImagePreprocessor input are vertically separated PIL images. Conversion to this form from the MNIST dataset make in prepare_mnist.load_data:

examples.utils.prepare_mnist
------------------------------------------------------

.. automodule:: examples.utils.prepare_mnist
   :members:
   :undoc-members:
   :show-inheritance:
