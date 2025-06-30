# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
from app.core.config import settings
from app.main import app
from create_test_token import create_token
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="function")
def mock_output_data_directory(request):
    """Create a mock output directory structure

    This fixture creates a mock application output directories inside the
    data root folder with the sample json outputs.

    Here is the mock output directory structure:
    /media/m2/output/
    └── app1
        ├── 20240813_232249
        │   └── nvflare_output.json
        ├── 20240813_202249
        │   └── nvflare_output.json
        ├── 20240812_232249
        │   └── nvflare_output.json
    └── app2
        ├── 20240813_232249
        │   └── nvflare_output.json
        ├── 20240813_202249
        │   └── nvflare_output.json
        ├── 20240812_232249
        │   └── nvflare_output.json
    └── app3
        ├── 20240813_232249
        │   └── nvflare_output.json
        ├── 20240813_202249
        │   └── nvflare_output.json
        ├── 20240812_232249
        │   └── nvflare_output.json
    """
    original_open = builtins.open
    with patch(request.param) as mock_data_path:
        # Mock the root directory path
        root_dir_path = MagicMock()
        root_dir_path.exists.return_value = True
        root_dir_path.is_dir.return_value = True
        root_dir_path.name = "output"

        # Mock three application folders inside the root directory
        app_dirs = ["app1", "app2", "app3"]
        app_dir_mocks = {}

        # Dictionary to store the content for each sample.json file
        file_contents = {}

        for dir in app_dirs:
            app_dir_mock = MagicMock()
            app_dir_mock.exists.return_value = True
            app_dir_mock.is_dir.return_value = True
            app_dir_mock.name = dir

            # Create 3 mock timestamp directories inside each app dir
            timestamp_dirs = {
                "20240813_232249": '{"Global": {"count": {"holoscan_set": {"feature1": 5000, "feature2": 7000}}}}',
                "20240813_202249": '{"Global": {"count": {"holoscan_set": {"feature1": 6000, "feature2": 8000}}}}',
                "20240812_232249": '{"Global": {"count": {"holoscan_set": {"feature1": 7000, "feature2": 9000}}}}',
            }
            stats_dirs = []
            for timestamp_dir, stats in timestamp_dirs.items():
                timestamp_dir_mock = MagicMock()
                timestamp_dir_mock.exists.return_value = True
                timestamp_dir_mock.is_dir.return_value = True
                timestamp_dir_mock.name = timestamp_dir

                # Mock the nvflare_output.json file
                output_file_mock = MagicMock()
                output_file_mock.exists.return_value = True
                output_file_mock.is_file.return_value = True
                output_file_mock.name = settings.stats_file_name

                content = json.dumps(stats)
                output_file_mock.read_text.return_value = content
                output_file_path = (
                    settings.data_root
                    + "/"
                    + app_dir_mock.name
                    + "/"
                    + timestamp_dir_mock.name
                    + "/"
                    + output_file_mock.name
                )
                file_contents[output_file_path] = content
                timestamp_dir_mock.iterdir.return_value = [output_file_mock]

                stats_dirs.append(timestamp_dir_mock)

            app_dir_mock.iterdir.return_value = stats_dirs
            app_dir_mocks[dir] = app_dir_mock

        root_dir_path.iterdir.return_value = app_dir_mocks.values()
        mock_data_path.return_value = root_dir_path

        def path_side_effect(arg):
            if arg == settings.data_root:
                return root_dir_path
            for folder, mock in app_dir_mocks.items():
                if arg == f"{settings.data_root}/{folder}":
                    return mock
            return MagicMock()

        mock_data_path.side_effect = path_side_effect

        # Patch the built-in open function to return different content for each file
        def selective_mock_open(file, mode="r", *args, **kwargs):
            file_str = str(file)
            if file_str in file_contents:
                mock_file = mock_open(read_data=file_contents[file_str]).return_value
                return mock_file
            else:
                # For files not in file_contents, use the real open function
                return original_open(file, mode, *args, **kwargs)

        with patch(
            "builtins.open",
            new_callable=lambda: MagicMock(side_effect=selective_mock_open),
        ):
            yield mock_data_path


@pytest.mark.parametrize(
    "mock_output_data_directory", ["app.api.v1.endpoints.get_apps.Path"], indirect=True
)
@pytest.mark.asyncio
async def test_get_apps(mock_output_data_directory):
    output_dir = mock_output_data_directory
    assert output_dir.exists()

    test_token = create_token()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test unauthorized access to a protected route
        test_route = f"{settings.API_V1_STR}/get_apps/"
        response = await ac.get(test_route)
        assert response.status_code == 401

        # Test authorized access to a protected route
        test_route = f"{settings.API_V1_STR}/get_apps/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_list = ["app1", "app2", "app3"]
        assert response.json() == expected_response_list


@pytest.mark.parametrize(
    "mock_output_data_directory",
    ["app.api.v1.endpoints.get_stats_list.Path"],
    indirect=True,
)
@pytest.mark.asyncio
async def test_get_stats_list(mock_output_data_directory):
    output_dir = mock_output_data_directory
    assert output_dir.exists()

    test_token = create_token()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test unauthorized access to a protected route
        test_route = f"{settings.API_V1_STR}/get_stats_list/app1/"
        response = await ac.get(test_route)
        assert response.status_code == 401

        # Test authorized access to a protected route
        test_route = f"{settings.API_V1_STR}/get_stats_list/app1/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_list = [
            "20240813_232249",
            "20240813_202249",
            "20240812_232249",
        ]
        assert response.json() == expected_response_list


@pytest.mark.parametrize(
    "mock_output_data_directory", ["app.api.v1.endpoints.get_stats.Path"], indirect=True
)
@pytest.mark.asyncio
async def test_get_stats(mock_output_data_directory):
    output_dir = mock_output_data_directory
    assert output_dir.exists()

    test_token = create_token()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test unauthorized access to a protected route
        test_route = f"{settings.API_V1_STR}/get_stats/app1/?timestamp=20240813_202249"
        response = await ac.get(test_route)
        assert response.status_code == 401

        # Test without providing a stats directory, should return the latest available stats.
        test_route = f"{settings.API_V1_STR}/get_stats/app1/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_stats_json = '{"Global": {"count": {"holoscan_set": {"feature1": 5000, "feature2": 7000}}}}'
        assert response.json() == expected_response_stats_json

        # Test with providing a stats directory, should return the stats from the given directory.
        test_route = f"{settings.API_V1_STR}/get_stats/app1/?timestamp=20240813_202249"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_stats_json = '{"Global": {"count": {"holoscan_set": {"feature1": 6000, "feature2": 8000}}}}'
        assert response.json() == expected_response_stats_json


@pytest.mark.parametrize(
    "mock_output_data_directory",
    ["app.api.v1.endpoints.get_range_stats.Path"],
    indirect=True,
)
@pytest.mark.asyncio
async def test_get_range_stats(mock_output_data_directory):
    output_dir = mock_output_data_directory
    assert output_dir.exists()

    test_token = create_token()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test unauthorized access to a protected route
        test_route = f"{settings.API_V1_STR}/get_range_stats/app1/20240812_232249/20240813_232249/"
        response = await ac.get(test_route)
        assert response.status_code == 401

        # Test range stats, should return accumulated stats for the given dates ranges
        test_route = f"{settings.API_V1_STR}/get_range_stats/app1/20240812_232249/20240813_232249/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_stats_json = {
            "Global": {
                "count": {"holoscan_set": {"feature1": 18000, "feature2": 24000}}
            }
        }
        assert response.json() == expected_response_stats_json

        # Test another range, should return accumulated stats for the given dates ranges
        test_route = f"{settings.API_V1_STR}/get_range_stats/app1/20240813_202249/20240813_232249/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_stats_json = {
            "Global": {
                "count": {"holoscan_set": {"feature1": 11000, "feature2": 15000}}
            }
        }
        assert response.json() == expected_response_stats_json

        # Specifying the same start and end timestamps should return stats for the given timestamp
        test_route = f"{settings.API_V1_STR}/get_range_stats/app1/20240813_202249/20240813_202249/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_stats_json = {
            "Global": {"count": {"holoscan_set": {"feature1": 6000, "feature2": 8000}}}
        }
        assert response.json() == expected_response_stats_json

        # Providing any random timestamps and not just from the available timestamps should also work
        test_route = f"{settings.API_V1_STR}/get_range_stats/app1/20230101_000000/20241231_000000/"
        response = await ac.get(
            test_route, headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        expected_response_stats_json = {
            "Global": {
                "count": {"holoscan_set": {"feature1": 18000, "feature2": 24000}}
            }
        }
        assert response.json() == expected_response_stats_json
