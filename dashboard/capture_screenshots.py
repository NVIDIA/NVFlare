# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the packaged Dashboard with synthetic data and an isolated Flask backend.

Run from the repository root: python -m dashboard.capture_screenshots
Install playwright==1.64.1 and its Chromium browser first, or select installed Chrome
with NVFLARE_TEST_BROWSER_CHANNEL=chrome. No existing Dashboard database is used.
"""

import argparse
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from dashboard.build_ui import check_build
from nvflare.dashboard.application import db, init_app
from nvflare.dashboard.application.models import Client, User
from nvflare.dashboard.application.store import Store
from nvflare.dashboard.config import Config

REPOSITORY = Path(__file__).resolve().parents[1]
NAMESPACE = "/nvflare-dashboard"
PASSWORD = "Screenshot-fixture-2026!"
PROJECT_TITLE = "Federated Learning Project"
PROJECT_SHORT_NAME = "Project"
VIEWPORT = {"width": 1600, "height": 1000}


def seed_project(app):
    with app.app_context():
        Store.set_project(
            {
                "short_name": PROJECT_SHORT_NAME,
                "title": PROJECT_TITLE,
                "description": (
                    "An example federated learning project.\n"
                    "Register your organization and client sites to participate."
                ),
                "app_location": "https://registry.example.test/project/training",
                "starting_date": "2026-11-10",
                "end_date": "2027-04-30",
                "server1": "server.example.test",
                "public": True,
            }
        )
        admin = User.query.filter_by(email="admin@example.test").one()
        admin.name = "Project Admin"
        for name, email, org, role, approval in (
            ("Org Admin A", "orgadmin-a@example.test", "OrgA", "org_admin", 100),
            ("Member A", "member-a@example.test", "OrgA", "member", 100),
            ("Lead B", "lead-b@example.test", "OrgB", "lead", 100),
            ("Org Admin B", "orgadmin-b@example.test", "OrgB", "org_admin", 100),
            ("Pending Member", "member-pending@example.test", "OrgB", "member", 0),
        ):
            result = Store.create_user(
                {"name": name, "email": email, "password": PASSWORD, "organization": org, "role": role}
            )
            assert result and result["status"] == "ok"
            user = User.query.filter_by(email=email).one()
            user.approval_state = approval
        db.session.commit()
        for name, owner, org, gpus, memory, approved in (
            ("site-a-1", "orgadmin-a@example.test", "OrgA", 4, 80, True),
            ("site-a-2", "orgadmin-a@example.test", "OrgA", 2, 40, False),
            ("site-b-1", "orgadmin-b@example.test", "OrgB", 4, 80, True),
        ):
            result = Store.create_client(
                {"name": name, "organization": org, "capacity": {"num_of_gpus": gpus, "mem_per_gpu_in_GiB": memory}},
                owner,
            )
            assert result and result["status"] == "ok"
            if approved:
                Store.patch_client_by_project_admin(result["client"]["id"], {"approval_state": 100})
        for model in (User, Client):
            for record in model.query.all():
                record.created_at = datetime(2026, 11, 5, 9, 0)
                record.updated_at = datetime(2026, 11, 5, 9, 0)
        db.session.commit()


def capture(browser, base_url, output):
    errors = []

    def new_page():
        context = browser.new_context(
            base_url=base_url, viewport=VIEWPORT, device_scale_factor=1, reduced_motion="reduce"
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        return page

    def shot(page, name):
        expect(
            page.get_by_role("banner", include_hidden=True).get_by_text(PROJECT_SHORT_NAME, exact=True)
        ).to_be_visible()
        page.evaluate("document.fonts.ready")
        expect(page.get_by_text("Loading your profile…", exact=True)).not_to_be_visible()
        page.mouse.move(1550, 950)
        page.screenshot(path=str(output / f"{name}.png"), full_page=True, animations="disabled", caret="hide")
        print(f"Captured {name}.png", flush=True)

    def login(page, email, destination):
        page.goto(NAMESPACE + "/")
        page.get_by_role("textbox", name="Email", exact=True).fill(email)
        page.get_by_label(re.compile("^Password")).fill(PASSWORD)
        page.get_by_role("button", name="Log in", exact=True).click()
        page.wait_for_url("**/" + destination)

    anonymous = new_page()
    anonymous.goto(NAMESPACE + "/")
    expect(anonymous.get_by_text(PROJECT_TITLE, exact=True)).to_be_visible()
    expect(anonymous.get_by_role("link", name="Sign Up", exact=True)).to_be_visible()
    shot(anonymous, "dashboard_login")

    applicant = new_page()
    applicant.goto(NAMESPACE + "/registration-form")
    applicant.get_by_label("Name", exact=True).fill("Pending Org Admin")
    applicant.get_by_label("Email", exact=True).fill("orgadmin-pending@example.test")
    applicant.get_by_label("Password", exact=True).fill(PASSWORD)
    applicant.get_by_label("Confirm password", exact=True).fill(PASSWORD)
    shot(applicant, "dashboard_user_registration")
    applicant.get_by_role("button", name="Register", exact=True).click()
    applicant.get_by_label("Organization", exact=True).fill("OrgB")
    applicant.get_by_role("combobox", name="Role", exact=True).click()
    applicant.get_by_role("option", name="Org Admin", exact=True).click()
    shot(applicant, "dashboard_user_registration_2")
    applicant.get_by_role("button", name="Save", exact=True).click()
    expect(
        applicant.get_by_text("Org Admin request pending approval: register your own client sites here.")
    ).to_be_visible()
    applicant.get_by_role("button", name="Add Site", exact=True).click()
    applicant.get_by_label("Site Name", exact=True).fill("site-b-2")
    applicant.get_by_label("Number of GPUs", exact=True).fill("2")
    applicant.get_by_label("Memory per GPU (GiB)", exact=True).fill("40")
    applicant.get_by_role("dialog").get_by_role("button", name="Save", exact=True).click()
    expect(applicant.get_by_role("dialog")).not_to_be_visible()
    expect(applicant.get_by_role("button", name="site-b-2", exact=True)).to_be_visible()
    shot(applicant, "dashboard_org_admin_reg_clients")
    applicant.get_by_role("button", name="Next", exact=True).click()
    applicant.get_by_role("button", name="Go to my dashboard", exact=True).click()
    expect(applicant.get_by_text("Requested role: Org Admin", exact=True)).to_be_visible()
    expect(applicant.get_by_role("button", name="site-b-2", exact=True)).to_be_visible()
    shot(applicant, "dashboard_org_admin_dashboard")

    member = new_page()
    login(member, "member-a@example.test", "user-dashboard")
    expect(member.get_by_text("Member A", exact=True)).to_be_visible()
    shot(member, "dashboard_user_info")

    admin = new_page()
    login(admin, "admin@example.test", "project-admin-dashboard")
    row = admin.get_by_role("row").filter(has_text="orgadmin-pending@example.test")
    expect(row).to_contain_text("Org Admin (requested)")
    row.get_by_role("checkbox").click()
    expect(row.get_by_role("checkbox")).to_be_checked()
    shot(admin, "dashboard_users_dashboard")
    admin.goto(NAMESPACE + "/site-dashboard")
    row = admin.get_by_role("row").filter(has_text="site-b-2")
    expect(row).to_contain_text("Pending")
    row.get_by_role("checkbox").click()
    expect(row.get_by_role("checkbox")).to_be_checked()
    shot(admin, "dashboard_sites_dashboard")
    admin.goto(NAMESPACE + "/project-configuration")
    expect(admin.get_by_label("Title", exact=True)).to_have_value(PROJECT_TITLE)
    shot(admin, "dashboard_project_configuration")
    admin.goto(NAMESPACE + "/server-config")
    expect(admin.get_by_label(re.compile("^Server hostname or IP address"))).to_have_value("server.example.test")
    shot(admin, "dashboard_server_config")
    admin.goto(NAMESPACE + "/")
    expect(admin.get_by_role("button", name="Freeze Project", exact=True)).to_be_enabled()
    admin.get_by_role("button", name="Freeze Project", exact=True).click()
    admin.get_by_role("dialog").get_by_role("button", name="Freeze", exact=True).click()
    expect(admin.get_by_text("Project status: Frozen", exact=True)).to_be_visible()

    member.goto(NAMESPACE + "/downloads")
    expect(member.get_by_role("button", name="Download FLARE Console", exact=True)).to_be_enabled()
    shot(member, "dashboard_member_download")
    owner = new_page()
    login(owner, "orgadmin-a@example.test", "user-dashboard")
    owner.goto(NAMESPACE + "/downloads")
    row = owner.get_by_role("row").filter(has_text="site-a-1")
    expect(row.get_by_role("button", name="Download Startup Kit", exact=True)).to_be_enabled()
    shot(owner, "dashboard_org_admin_downloads")
    with owner.expect_download() as download:
        row.get_by_role("button", name="Download Startup Kit", exact=True).click()
    assert download.value.failure() is None
    expect(owner.get_by_role("dialog")).to_contain_text("The secure PIN is:")
    shot(owner, "dashboard_download_pin")
    admin.goto(NAMESPACE + "/downloads")
    expect(admin.get_by_role("button", name="Download Server Startup Kit", exact=True)).to_be_enabled()
    shot(admin, "dashboard_proj_admin_downloads")
    assert not errors, "Unhandled browser errors: " + str(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "docs" / "resources")
    parser.add_argument("--browser-channel", default=os.environ.get("NVFLARE_TEST_BROWSER_CHANNEL"))
    args = parser.parse_args()
    check_build(REPOSITORY / "dashboard" / "ui", REPOSITORY / "nvflare/dashboard/application/static/nvflare-dashboard")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="nvflare-dashboard-screenshots-") as folder:
        os.environ["NVFL_WEB_ROOT"] = folder
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(folder) / 'db.sqlite'}"
        os.environ["NVFL_CREDENTIAL"] = f"admin@example.test:{PASSWORD}:ProjectOrg"
        os.environ["NVFL_DASHBOARD_STATIC_FOLDER"] = str(REPOSITORY / "nvflare/dashboard/application/static")
        Config.SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
        Config.SECRET_KEY = "isolated-synthetic-dashboard-screenshot-key-2026"
        app = init_app()
        seed_project(app)
        server = make_server("127.0.0.1", 0, app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as manager:
                browser = manager.chromium.launch(channel=args.browser_channel)
                try:
                    capture(browser, f"http://127.0.0.1:{server.server_port}", args.output_dir)
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            with app.app_context():
                db.session.remove()
                db.engine.dispose()
            if thread.is_alive():
                raise RuntimeError("Screenshot backend did not stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
