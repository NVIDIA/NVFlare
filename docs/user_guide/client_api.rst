.. _client_api:

##########
Client API
##########

NVFlare Client API provides an easy way for users to convert their centralized, local
training code into a federated learning code.

It brings the following benefits:

* Enable a quicker start by reducing the number of new NVFlare specific concepts
   a user has to learn when first working with Federated Learning using NVFlare.

* Enable easy adaptation from existing local training code using different framework
   (pytorch, pytorch lightning, huggingface) to run the application in a
   federated setting by just few lines of code changes

************
Core concept
************

Federated learning's concept is for each participating site to get a good model (better than
locally trained model) without sharing the data.

It is done by sharing model parameters or parameter differences (certain filters can be used to
ensure privacy-preserving and protects against gradient inversion attacks) to each other.

The aggregators will take in all these model parameters submitted by each site and produce a
new global model.

We hope that this new global model will be better than locally trained model since it
conceptually trained on more data.

One of the popular federated learning workflow, "FedAvg" is like this:

#. Server site initialize an initial model
#. For each round:

   #. server sends the global model to clients
   #. each client starts with this global model and train on their own data
   #. each client sends back their trained model
   #. server aggregates all the models and produces a new global model

On the client side, the training workflow is:

#. get a model from server side
#. local training
#. send a trained model to server side

To be able to support different training frameworks, we define a standard data structure called "FLModel"
for the local training code to exchange information with NVFlare system.

We explain its attributes below:

.. literalinclude:: ../../nvflare/app_common/abstract/fl_model.py
   :language: python
   :lines: 41-67
   :linenos:
   :caption: fl_model.py

Users only need to get the required information from this data structure,
run local training, and put the results back into this data structure to be aggregated on the aggregator side.


For a general use case, there are three essential methods for the Client API:

* `init()`: Initializes NVFlare Client API environment.
* `receive()`: Receives model from NVFlare side.
* `send()`: Sends the model to NVFlare side.


Users can use these APIs to change their centralized training code to federate learning, for example:

.. code-block:: python

    import nvflare.client as flare

    flare.init()
    input_model = flare.receive()
    new_params = local_train(input_model.params)
    output_model = flare.FLModel(params=new_params)
    flare.send(output_model)

Please refer to (:mod:`nvflare.client` for all the APIs)

For more examples of using Client API with different frameworks,
please refer to `examples/hello-world/ml-to-fl <https://github.com/NVIDIA/NVFlare/tree/main/examples/ml-to-fl>`_.
