Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` provides a
command line interface (CLI) with two primary subcommands.

-------------------------------
zea — main entry point
-------------------------------

The ``zea`` command exposes two subcommands:

.. code-block:: text

    zea process <dataset> <save_dir> [options]   # batch beamform a dataset
    zea app [--share] [--server_port PORT]        # launch the Gradio visualizer

.. autoprogram:: zea.__main__:get_parser()
   :prog: zea

--------------------------------
Process dataset (standalone CLI)
--------------------------------

The beamformer can also be invoked directly as a module (equivalent to ``zea process``):

.. autoprogram:: zea.data.process:get_parser()
   :prog: python -m zea.data.process

-------------------------------
Convert datasets
-------------------------------

.. autoprogram:: zea.data.convert.__main__:get_parser()
   :prog: python -m zea.data.convert

-------------------------------
Data copying
-------------------------------

.. autoprogram:: zea.data.__main__:get_parser()
   :prog: python -m zea.data

.. _cli-file-operations:

-------------------------------
Data file manipulation
-------------------------------

.. autoprogram:: zea.data.file_operations:get_parser()
   :prog: python -m zea.data.file_operations
