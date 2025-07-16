# Visualization backend for Holoscan Federated Analytics

This component functions as the visualization backend for Holoscan Federated Analytics applications. It offers the necessary REST APIs to deliver global statistics to the visualization frontend. The REST API service is built using Python and the [FastAPI](https://fastapi.tiangolo.com/) framework. Authorization and authentication are managed by the visualization frontend. At present, APIs are safeguarded with an encoded JWT, and a simple script is provided to generate the JWT needed for each REST API call.


# Building the backend REST API service container

```bash
    cd <src_root>/visualization/backend/
    docker compose build
```


# Running the backend container

```bash
    cd <src_root>/visualization/backend/
    docker compose run
```
The REST API service will be accessible on port `8000` by default, as indicated in the `docker-compose.yml` configuration file.


# Validating the backend container / REST API service

The REST API service can be evaluated using the `curl` utility. Follow the instructions below to test different APIs.

1. Run the Holoscan application with analytics capabilities to generate the data required for the Holoscan Federated Analytics application.

2. Run the NVFLARE application for Holoscan Federated Analytics to generate the global statistics. Follow steps mentioned [here](../../applications/holoscan_nvflare_app/README.md).

3. Start the backend container using the steps outlined above. This can be performed on any of the IGX systems where Holoscan applications are operational or on any other Linux-based system.

4. Create the test JWT token that can be used to authenticate REST API service.

    Open another terminal and run below commands:

    ```bash
        cd <src_root>/visualization/backend/
        python3 -m venv .venv
        source .venv/bin/activate
        pip3 install -r requirements.txt
        TOKEN=`python3 ./create_test_token.py`
    ```

5. Modify the `DATA_ROOT` in [.env](../backend/app/.env) file to point the the root directory used for Holoscan applications analytics data. By default, it is `/media/m2/output`.

6. Getting the list of registered applications (API: `/get_apps/`)

   Run the command below from the terminal where test JWT was created and `TOKEN` variable was set in step #3. It should output a list containing the application names. If analytics results are available for a single application, the returned list will have just one entry.

    ```bash
        curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/get_apps/
    ```

7. Getting the list of available stats for a given application (API: `/get_stats_list/{app_name}/`)

   During each execution of an analytics-enabled application, a distinct directory is generated within the application directory to store the statistics output. The API `/get_stats_list` returns the list of available statistics for a specific application. The example command below will display the list of statistics available for the application named `out_of_body_detection`. The outcome is a list of timestamps.

    ```bash
        curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/get_stats_list/out_of_body_detection/
    ```

8. Getting the statistics for the given application (API: `/get_stats/{app_name}/`)

    Example command below to get the latest available statistics for the application named `out_of_body_detection`
    ```bash
        curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/get_stats_list/out_of_body_detection/
    ```

    And below command would return the statistics for a given date (timestamp) for the application named `out_of_body_detection`
    ```bash
        curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/get_stats/out_of_body_detection/?timestamp=20240716_153507
   ```

9. Getting the statistics for the date range (API: `/get_range_stats/{app_name}/{start}/{end}/`)

    Below command would return the accumulated statistics for a given date range for the application named `out_of_body_detection`
    ```bash
        curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/get_stats/out_of_body_detection/20240716_153507/20240717_150000/
   ```

# Unit Testing

```bash
cd <src_root>/visualization/backend/
pip3 install -r requirements.txt
pytest
```