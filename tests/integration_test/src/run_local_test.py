
from tests.integration_test.src.nvf_test_driver_flare_api import NVFFLAREAPITestDriver
from tests.integration_test.src.utils import check_job_done_FLARE_API


if __name__ == "__main__":
    test_driver = NVFFLAREAPITestDriver(
    download_root_dir="/tmp/tmp_a5jpli4/admin/transfer", site_launcher=None, poll_period=1
        )
    test_driver.initialize_super_user(
        workspace_root_dir="/tmp/tmp_a5jpli4", upload_root_dir="/workspace/repos/NVFlare_experiment2/tests/integration_test/data/apps", poc=True, super_user_name="admin"
    )
    
    print(check_job_done_FLARE_API(job_id="f113af4e-b5bf-41f1-b773-a85a7c2d64af", admin_api=test_driver.super_admin_api))