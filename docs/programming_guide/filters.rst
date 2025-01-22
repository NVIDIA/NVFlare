.. _filters:

#######
Filters
#######
Filters in NVIDIA FLARE are a type of FLComponent that has a ``process`` method to transform the ``Shareable`` object between
the communicating parties. A ``Filter`` can be used to provide additional processing to shareable data before sending or
after receiving from the peer.

The ``FLContext`` is available for the ``Filter`` to use.

.. literalinclude:: ../../nvflare/apis/filter.py
    :language: python
    :lines: 22-

In config_fed_server.json and config_fed_client.json (for details see :ref:`application`),
task_result_filters and task_data_filters can be configured for processing data at the points
highlighted in the image below:

.. image:: ../resources/Filters.png
    :height: 350px

.. _dxo_based_filtering:

****************************************
DXO Based Filtering
****************************************

In NVFLARE, filters are used for the pre and post processing of a task. 

On the Server side, before sending the task to the Client, "task data filters" (if any) are applied to the task data. Only the filtered task data is sent to the client. Similarly, when the task result is received from the client, "task result filters" are applied to the received result before passing on to the Controller.

On the Client side, once a task is received from the Server, "task data filters" (if any) are applied to the task data before passing to the task executor. Similarly, when the task result is computed from the executor, "task result filters" are applied to the task result before sending it to the Server.

Filters are the primary technique for data privacy protection.

.. _filters_for_privacy:

Filters can convert data formats and a lot more. You can apply any type of massaging to the data for the
purpose of security. In fact, privacy and homomorphic encryption techniques are all implemented as filters:

    - ExcludeVars to exclude variables from shareable (:mod:`nvflare.app_common.filters.exclude_vars`)
    - PercentilePrivacy for truncation of weights by percentile (:mod:`nvflare.app_common.filters.percentile_privacy`)
    - SVTPrivacy for differential privacy through sparse vector techniques (:mod:`nvflare.app_common.filters.svt_privacy`)
    - Homomorphic encryption filters to encrypt data before sharing (:mod:`nvflare.app_common.homomorphic_encryption.he_model_encryptor.py` and :mod:`nvflare.app_common.homomorphic_encryption.he_model_decryptor`)

For an example application using SVTPrivacy, see :github_nvflare_link:`Differential Privacy for BraTS18 segmentation (GitHub) <examples/advanced/brats18>`.

DXO - Data Exchange Object
===========================
The message object passed between the server and clients is of the Shareable class. Shareable is a general structure for all kinds of communication (task interaction, aux messages, fed events, etc.) that in addition to the message payload, also carries contextual information (such as peer FL context). NVFLARE's DXO object is a general-purpose structure that is meant to be used to carry message payload in a self-descriptive manner. As an analogy, think of Shareable as an HTTP message, whereas a DXO as a JPEG image that is carried by the HTTP message.

A DXO object has the following properties:

    - Data Kind - the kind of data the DXO object carries (e.g. WEIGHTS, WEIGHT_DIFF, COLLECTION of DXOs, etc.)
    - Meta - meta properties that describe the data (e.g. whether processed/encrypted and processing algorithm). This is a dict.
    - Data - the dict that holds data of the DXO. 

Note that a DXO object could be of COLLECTION kind. In this case, the Data of the DXO is a dict of DXO objects.

DXO Filter
==========
Even though a filter can be written to process anything in a Shareable, for data privacy processing, filtering is usually against the payload itself. DXO-based filters could be very useful when the payload is a DXO object.

DXOFilter is a subclass of Filter and a mini-framework that makes it easy to write DXO-based filters. To write a DXO-based filter, you create the filter as a subclass of DXOFilter. Instead of writing the "process" method of the Filter class, you will write the "process_dxo" method. The "process" method is provided by the DXOFilter class.

Your subclass of DXOFilter benefits from the features of DXOFilter:

    - DXO structure processing. Since a DXO could contain a collection of sub-DXOs, which can contain even more DXOs, you can view the whole DXO as a tree of DXO nodes. Traversing this tree is done for you by the DXOFilter's "process" method.
    - Data Kind checking. Your process_dxo method is called to process a DXO only when the DXO node is a data kind that your filter is configured to handle.
    - Filtering history recording. If a DXO node is processed by your filter, your filter's class name will be appended to the DXO's "filter_history"
    - Auditing. If your filter is applied, a job audit event will be created to record the fact that the filter is applied to data.

Creating a DXO Filter
---------------------
You create a new DXO-based filter by extending the DXOFilter class, and provide the "process_dxo" method.

In your constructor, you need to determine supported DXO kinds and make them known to the super class (supported_data_kinds). You also need to specify what data kinds your filter is to be applied to (data_kinds_to_filter). Of course, data_kinds_to_filter must be a subset of the supported_data_kinds. Typically data_kinds_to_filter should be user configurable.
Specifying supported_data_kinds makes it clear what the filter is capable of, and data_kinds_to_filter specifies how this particular filter is used.

Pay attention to the return value of the "process_dxo" method that you will write. You must return None if no processing is done to the DXO object passed to you. You must return a DXO object (could be the same DXO passed to you or a newly created one) if processing is applied to the DXO passed to you.

In the past, filters were written with implicitly assumed data kinds. They did not explicitly specify what kinds of data they can process. This worked sort of okay because filters could only be specified in the job configuration by a researcher, who usually knows what filters are applicable to the job. But this won't work for site privacy policy where specified filters are for all jobs. DXO based filters work in a different way now: instead of assuming the data is always to be processed, DXOFilter only filters the DXO objects that are configured to be processed based on their data kinds - if a DXO object is not a configured kind, then it won't be processed. This makes it possible for the Org Admin to simply specify filters based on data kinds they want to control.
