.. _filters:

Filters
=======
Filters in NVIDIA FLARE are a type of FLComponent that has a ``process`` method to transform the ``Shareable`` object between
the communicating parties. A ``Filter`` can be used to provide additional processing to shareable data before sending or
after receiving from the peer.

The ``FLContext`` is available for the ``Filter`` to use.

.. literalinclude:: ../../nvflare/apis/filter.py
    :language: python
    :lines: 22-

In config_fed_server.json and config_fed_client.json (for details see :ref:`user_guide/application:NVIDIA FLARE Application`),
task_result_filters and task_data_filters can be configured for processing data at the points
highlighted in the image below:

.. image:: ../resources/Filters.png
    :height: 350px

.. _filters_for_privacy:

Filters can convert data formats and a lot more. You can apply any type of massaging to the data for the
purpose of security. In fact, privacy and homomorphic encryption techniques are all implemented as filters:

    - ExcludeVars to exclude variables from shareable (:mod:`nvflare.app_common.filters.exclude_vars`)
    - PercentilePrivacy for truncation of weights by percentile (:mod:`nvflare.app_common.filters.percentile_privacy`)
    - SVTPrivacy for differential privacy through sparse vector techniques (:mod:`nvflare.app_common.filters.svt_privacy`)
    - Homomorphic encryption filters to encrypt data before sharing (:mod:`nvflare.app_common.homomorphic_encryption.he_model_encryptor.py` and :mod:`nvflare.app_common.homomorphic_encryption.he_model_decryptor`)
