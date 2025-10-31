.. _data_privacy_protection:

Data Privacy Protection
=======================
Federated learning activities are performed with task-based interactions between the server and FL clients: the server
issues tasks to the clients, and clients process tasks and return results back to the server. NVFLARE comes with a
general-purpose data :ref:`filtering mechanism <filters>` for processing task data and results:

    - On the Server: before task data is sent to the client, the configured "task_data_filters" defined in the job are executed.
    - On the Client: when the task data is received by the client and before giving it to the executor for processing, NVFLARE framework applies configured "task_data_filters" defined in the job.
    - On the Client: after the execution of the task by the executor and before sending the produced result back to the server, NVFLARE framework applies configured "task_result_filters" to the result before sending to the Server.
    - On the Server: after receiving the task result from the client, the NVFLARE framework applies the configured "task_result_filters" before giving it to the controller for processing.

This mechanism has been used for the purpose of data privacy protection on the client side. For example, differential
privacy filters can be applied to model weights before sending to the server for aggregation.

NVFLARE has implemented github_nvflare_link:`some commonly used privacy protection filters. <nvflare/app_common/filters>`
