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

"""Exercise the shipped Dashboard pages against a live backend.

Optional browser setup: pip install playwright && playwright install chromium.
Set NVFLARE_TEST_BROWSER_CHANNEL=chrome to use an installed Google Chrome.
Run: pytest tests/integration_test/fast/dashboard_validation_ui_test.py -v
"""

import io
import json
import os
import re
import threading
from uuid import uuid4
from zipfile import ZipFile

import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID
from werkzeug.serving import make_server

from nvflare.dashboard.application import db
from nvflare.dashboard.application.models import Client, Project, User
from nvflare.dashboard.application.store import Store
from tests.dashboard_test_utils import TEST_PW, TEST_USER

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
NS = "/nvflare-dashboard"
API = NS + "/api/v1"
ORG_ERROR = "Organization must contain only letters, digits, and underscores."


@pytest.fixture(scope="session")
def app(dashboard_app):
    return dashboard_app


@pytest.fixture(scope="module")
def live_dashboard(app):
    with app.app_context():
        project = Project.query.first()
        original = {column.name: getattr(project, column.name) for column in Project.__table__.columns}
        Store.set_project({"short_name": "ui_validation", "server1": "localhost", "public": True})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield app, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        with app.app_context():
            project = Project.query.first()
            for name, value in original.items():
                setattr(project, name, value)
            db.session.commit()
        assert not thread.is_alive(), "Dashboard server did not stop"


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as manager:
        browser = manager.chromium.launch(channel=os.environ.get("NVFLARE_TEST_BROWSER_CHANNEL"))
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(browser, live_dashboard):
    _, base_url = live_dashboard
    context = browser.new_context(base_url=base_url)
    page = context.new_page()
    failures = []
    page.on("pageerror", lambda error: failures.append(str(error)))
    try:
        yield page
    finally:
        context.close()
    assert failures == [], "Unhandled browser errors: " + str(failures)


def login(page, email, password, role):
    response = page.request.post(API + "/login", data={"email": email, "password": password})
    assert response.status == 200
    data = response.json()
    session = {
        "user": {"email": email, "token": data["access_token"], "id": data["user"]["id"], "role": role},
        "status": "authenticated",
    }
    page.add_init_script(f"localStorage.setItem('session', {json.dumps(json.dumps(session))})")


def validation_dialog(page, action, message):
    # Accept immediately: a native alert can block the triggering click/keypress.
    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_event("dialog") as dialog_info:
        action()
    assert dialog_info.value.message == message


def test_registration_invalid_organization_can_be_corrected(page, live_dashboard):
    app, _ = live_dashboard
    email = f"registration-{uuid4().hex}@test.com"
    page.goto(NS + "/registration-form")
    page.locator('input[name="email"]').fill(email)
    page.locator('input[name="name"]').fill("Test User")
    page.locator('input[name="password"]').fill("Testing1234!")
    page.locator('input[name="confirm_password"]').fill("Testing1234!")
    page.get_by_role("button", name="Register", exact=True).click()
    # Registration does not advance until automatic login has completed.
    page.wait_for_function("JSON.parse(localStorage.getItem('session') || '{}').user?.token")
    organization = page.locator('input[name="org"]')
    expect(organization).to_be_visible()
    organization.fill("Test Holger Client")
    with page.expect_response(
        lambda response: response.request.method == "PATCH" and "/users/" in response.url
    ) as saved:
        page.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.status == 400
    expect(page.get_by_text(ORG_ERROR, exact=True)).to_be_visible()
    expect(organization).to_be_enabled()
    expect(organization).to_have_value("Test Holger Client")
    with app.app_context():
        assert User.query.filter_by(email=email).one().organization.name == ""
    organization.fill("Test_Holger_Client")
    with page.expect_response(
        lambda response: response.request.method == "PATCH" and "/users/" in response.url
    ) as saved:
        page.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.status == 200
    expect(organization).not_to_be_visible()
    with app.app_context():
        user = User.query.filter_by(email=email).one()
        assert user.organization.name == "Test_Holger_Client"
        Store.delete_user(user.id)


def test_server_name_error_keeps_form_editable(page):
    login(page, TEST_USER, TEST_PW, 4)
    page.goto(NS + "/server-config")
    server = page.locator('input[name="server1"]')
    server.fill("bad host")
    with page.expect_response(
        lambda response: response.request.method == "PATCH" and response.url.endswith("/project")
    ) as saved:
        page.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.status == 400
    expect(
        page.get_by_text("Server name must be a valid hostname or IPv4 address, without a scheme or port.", exact=True)
    ).to_be_visible()
    expect(server).to_be_enabled()
    server.fill("server.test")
    with page.expect_response(
        lambda response: response.request.method == "PATCH" and response.url.endswith("/project")
    ) as saved:
        page.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.status == 200
    expect(server).to_be_enabled()


@pytest.mark.parametrize("resource", ["users", "clients", "servers"])
def test_invalid_legacy_download_shows_error_and_allows_retry(page, live_dashboard, resource):
    app, _ = live_dashboard
    email = f"download-{uuid4().hex}@test.com"
    with app.app_context():
        project = Project.query.first()
        Store.set_project({"frozen": True})
        created = Store.create_user(
            {"email": email, "password": "test", "organization": "Download_org", "role": "org_admin"},
            created_by_project_admin=True,
        )["user"]
        user = User.query.get(created["id"])
        user.approval_state = 100
        if resource == "clients":
            created_site = Store.create_client({"name": "download-site", "organization": "Download_org"}, email)[
                "client"
            ]
            target = Client.query.get(created_site["id"])
            target.approval_state = 100
            attribute, invalid, button, message = (
                "name",
                "bad name",
                "Download Startup Kit",
                "Client name must contain only letters, digits, hyphens, and underscores.",
            )
        elif resource == "users":
            target = user.organization
            attribute, invalid, button, message = "name", "Bad org", "Download FLARE Console", ORG_ERROR
        else:
            target = project
            attribute, invalid, button, message = (
                "server1",
                "bad host",
                "Download Server Startup Kit",
                "Server name must be a valid hostname or IPv4 address, without a scheme or port.",
            )
        original = getattr(target, attribute)
        setattr(target, attribute, invalid)
        db.session.commit()
        try:
            login(
                page,
                TEST_USER if resource == "servers" else email,
                TEST_PW if resource == "servers" else "test",
                4 if resource == "servers" else 3,
            )
            page.goto(NS + "/downloads")
            validation_dialog(page, lambda: page.get_by_role("button", name=button, exact=True).click(), message)
            expect(page.get_by_text("Your download should be starting soon.", exact=True)).not_to_be_visible()
            setattr(target, attribute, original)
            db.session.commit()
            with page.expect_download() as download:
                page.get_by_role("button", name=button, exact=True).click()
            assert download.value.suggested_filename.endswith(".zip")
            assert download.value.failure() is None
        finally:
            setattr(target, attribute, original)
            project.frozen = False
            db.session.commit()
            Store.delete_user(user.id)


@pytest.mark.parametrize("page_name,as_admin", [("user-dashboard", False), ("site-dashboard", True)])
def test_invalid_client_name_can_be_corrected(page, live_dashboard, page_name, as_admin):
    app, _ = live_dashboard
    email = f"site-owner-{uuid4().hex}@test.com"
    name = f"site-{uuid4().hex}"
    with app.app_context():
        user = Store.create_user(
            {"email": email, "password": "test", "organization": "Site_org", "role": "org_admin"},
            created_by_project_admin=True,
        )["user"]
        Store.patch_user_by_project_admin(
            user["id"], {"approval_state": 100, "reviewed_role": "org_admin", "reviewed_organization": "Site_org"}
        )
        site = Store.create_client({"name": name, "organization": "Site_org", "capacity": {"num_of_gpus": 4}}, email)[
            "client"
        ]
    try:
        login(page, TEST_USER if as_admin else email, TEST_PW if as_admin else "test", 4 if as_admin else 3)
        page.goto(NS + "/" + page_name)
        page.get_by_text(name, exact=True).click()
        editor = page.locator('input[title="name"]')
        editor.fill("bad name")
        validation_dialog(
            page,
            lambda: editor.press("Tab"),
            "Client name must contain only letters, digits, hyphens, and underscores.",
        )
        with app.app_context():
            assert Client.query.get(site["id"]).name == name
        if not as_admin:
            expect(editor).to_have_value("bad name")
            # Leaving a rejected edit must show the saved value in the table.
            editor.fill("")
            page.get_by_text("4", exact=True).last.click()
        expect(page.get_by_text(name, exact=True)).to_be_visible()
        page.reload()
        page.get_by_text(name, exact=True).click()
        expect(editor).to_be_enabled()
        editor.fill(name + "_fixed")
        with page.expect_response(
            lambda response: response.request.method == "PATCH" and "/clients/" in response.url
        ) as saved:
            editor.press("Tab")
        assert saved.value.status == 200
        expect(page.get_by_text(name + "_fixed", exact=True)).to_be_visible()
    finally:
        with app.app_context():
            Store.delete_user(user["id"])


def test_admin_rejected_organization_edit_preserves_saved_value(page, live_dashboard):
    app, _ = live_dashboard
    email = f"admin-edit-{uuid4().hex}@test.com"
    organization = "Org_" + uuid4().hex
    with app.app_context():
        user = Store.create_user({"email": email, "password": "test", "organization": organization})["user"]
    try:
        login(page, TEST_USER, TEST_PW, 4)
        page.goto(NS + "/project-admin-dashboard")
        page.get_by_text(organization, exact=True).click()
        editor = page.locator('input[title="organization"]')
        editor.fill("Bad org")
        validation_dialog(page, lambda: editor.press("Tab"), ORG_ERROR)
        expect(page.get_by_text(organization, exact=True)).to_be_visible()
        page.get_by_text(organization, exact=True).click()
        editor.fill(organization + "_fixed")
        with page.expect_response(
            lambda response: response.request.method == "PATCH" and "/users/" in response.url
        ) as saved:
            editor.press("Tab")
        assert saved.value.status == 200
        expect(page.get_by_text(organization + "_fixed", exact=True)).to_be_visible()
    finally:
        with app.app_context():
            Store.delete_user(user["id"])


def test_invalid_new_client_can_be_corrected(page, live_dashboard):
    app, _ = live_dashboard
    email = f"new-site-owner-{uuid4().hex}@test.com"
    name = "site-" + uuid4().hex
    with app.app_context():
        user = Store.create_user(
            {
                "email": email,
                "password": "test",
                "organization": "New_site_org",
                "role": "org_admin",
            },
            created_by_project_admin=True,
        )["user"]
    try:
        login(page, email, "test", 3)
        page.goto(NS + "/user-dashboard")
        page.get_by_role("button", name="Add Site", exact=True).click()
        editor = page.locator('input[title="name"]')
        editor.fill("bad name")
        validation_dialog(
            page,
            lambda: editor.press("Tab"),
            "Client name must contain only letters, digits, hyphens, and underscores.",
        )
        expect(editor).to_have_value("bad name")
        expect(editor).to_be_enabled()
        with app.app_context():
            assert Client.query.filter_by(name="bad name").first() is None
        editor.fill(name)
        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/clients")
        ) as saved:
            editor.press("Tab")
        assert saved.value.status == 201
        expect(page.get_by_text(name, exact=True)).to_be_visible()
        # A rejected later edit must not overwrite the saved table value either.
        page.get_by_text(name, exact=True).click()
        editor.fill("bad name")
        validation_dialog(
            page,
            lambda: editor.press("Tab"),
            "Client name must contain only letters, digits, hyphens, and underscores.",
        )
        expect(editor).to_have_value("bad name")
        editor.fill("")
        page.get_by_text("4", exact=True).last.click()
        expect(page.get_by_text(name, exact=True)).to_be_visible()
        with app.app_context():
            assert Client.query.filter_by(name=name).one().creator_id == user["id"]
    finally:
        with app.app_context():
            Store.delete_user(user["id"])


def register_through_ui(page, email):
    page.goto(NS + "/registration-form")
    page.locator('input[name="email"]').fill(email)
    page.locator('input[name="name"]').fill("Workshop Participant")
    page.locator('input[name="password"]').fill("Testing1234!")
    page.locator('input[name="confirm_password"]').fill("Testing1234!")
    page.get_by_role("button", name="Register", exact=True).click()
    expect(page.locator('input[name="org"]')).to_be_visible()
    page.locator('input[name="org"]').fill("Workshop_org")
    page.get_by_text("Member", exact=True).click()
    page.get_by_text("Org Admin", exact=True).click()


def approve_through_ui(page, identifier, action, resource):
    row = page.get_by_role("row").filter(has_text=identifier)
    row.get_by_role("checkbox").check()
    page.get_by_role("button", name="Bulk Actions", exact=True).click()
    page.get_by_role("button", name=action, exact=True).click()
    with page.expect_response(
        lambda response: response.request.method == "PATCH" and f"/{resource}/" in response.url
    ) as approved:
        page.get_by_role("button", name="Apply", exact=True).click()
    assert approved.value.status == 200, approved.value.text()
    expect(row).to_contain_text("Approved")


@pytest.mark.parametrize("frozen_at_registration", [False, True])
def test_org_admin_signup_sites_approval_and_encrypted_download(
    page, browser, live_dashboard, tmp_path, frozen_at_registration
):
    app, base_url = live_dashboard
    email = f"workshop-{uuid4().hex}@test.com"
    site_name = "workshop-" + uuid4().hex
    admin_context = browser.new_context(base_url=base_url)
    admin = admin_context.new_page()
    try:
        with app.app_context():
            project = Project.query.first()
            project.frozen = False
            db.session.commit()
            if frozen_at_registration:
                Store.set_project({"frozen": True})
        register_through_ui(page, email)
        with page.expect_response(
            lambda response: response.request.method == "PATCH" and "/users/" in response.url
        ) as saved:
            page.get_by_role("button", name="Save", exact=True).click()
        assert saved.value.status == 200
        expect(
            page.get_by_text("Org Admin request pending approval: register your own client sites here.", exact=True)
        ).to_be_visible()
        page.get_by_role("button", name="Add Site", exact=True).click()
        editor = page.locator('input[title="name"]')
        editor.fill(site_name)
        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/clients")
        ) as created:
            editor.press("Tab")
        assert created.value.status == 201
        assert created.value.json()["client"]["approval_state"] == 0
        page.get_by_role("button", name="Next", exact=True).click()
        page.get_by_role("button", name="Go to my dashboard", exact=True).click()
        expect(page.get_by_text(site_name, exact=True)).to_be_visible()
        expect(page.get_by_text("Requested role: Org Admin", exact=True)).to_be_visible()
        page.goto(NS + "/downloads")
        expect(page.get_by_role("button", name="Download Startup Kit", exact=True)).to_be_disabled()
        expect(page.get_by_role("button", name="Download FLARE Console", exact=True)).to_be_disabled()

        login(admin, TEST_USER, TEST_PW, 4)
        admin.goto(NS + "/project-admin-dashboard")
        row = admin.get_by_role("row").filter(has_text=email)
        expect(row).to_contain_text("Org Admin (requested)")
        expect(row).to_contain_text("Workshop_org")
        expect(row).to_contain_text("Pending")
        approve_through_ui(admin, email, "Approve User", "users")
        admin.goto(NS + "/site-dashboard")
        approve_through_ui(admin, site_name, "Approve Client", "clients")
        if not frozen_at_registration:
            admin.goto(NS + "/")
            # Use the actual project UI's freeze action, including its confirmation dialog.
            admin.get_by_role("button", name="Freeze Project", exact=True).click()
            with admin.expect_response(
                lambda response: response.request.method == "PATCH" and response.url.endswith("/project")
            ) as frozen:
                admin.get_by_role("button", name="Freeze", exact=True).click()
            assert frozen.value.status == 200
        page.reload()
        expect(page.get_by_role("button", name="Download Startup Kit", exact=True)).to_be_enabled()
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download Startup Kit", exact=True).click()
        assert downloaded.value.failure() is None
        kit_path = tmp_path / "site.zip"
        downloaded.value.save_as(kit_path)
        # The modal displays the PIN alongside the download confirmation.
        body = page.locator("body").inner_text()
        pin_match = re.search(r"The secure PIN is:\s*([A-Za-z0-9]+)", body)
        assert pin_match, body
        with ZipFile(kit_path) as kit:
            assert kit.getinfo("startup/fed_client.json").flag_bits & 1
            config = json.loads(kit.read("startup/fed_client.json", pwd=pin_match.group(1).encode()))
            assert config["client"]["connection_security"] == "mtls"
            certificate = x509.load_pem_x509_certificate(
                kit.read("startup/client.crt", pwd=pin_match.group(1).encode())
            )
            assert certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == site_name
        page.get_by_role("button", name="Okay", exact=True).click()
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download FLARE Console", exact=True).click()
        console_path = tmp_path / "console.zip"
        downloaded.value.save_as(console_path)
        body = page.locator("body").inner_text()
        pin_match = re.search(r"The secure PIN is:\s*([A-Za-z0-9]+)", body)
        assert pin_match, body
        with ZipFile(io.BytesIO(console_path.read_bytes())) as kit:
            certificate = x509.load_pem_x509_certificate(
                kit.read("startup/client.crt", pwd=pin_match.group(1).encode())
            )
            assert certificate.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)[0].value == "org_admin"
    finally:
        admin_context.close()
        with app.app_context():
            project = Project.query.first()
            project.frozen = False
            db.session.commit()
            user = User.query.filter_by(email=email).first()
            if user:
                Store.delete_user(user.id)


@pytest.mark.parametrize("status", [200, 403, 409, 401])
def test_role_rejection_keeps_registration_editable(page, live_dashboard, status):
    app, _ = live_dashboard
    email = f"rejected-{uuid4().hex}@test.com"
    register_through_ui(page, email)
    message = "Please review your requested role with the project administrator."
    page.route(
        "**/api/v1/users/*", lambda route: route.fulfill(status=status, json={"status": "error", "message": message})
    )
    try:
        page.get_by_role("button", name="Save", exact=True).click()
        expected = "Your session has expired. Please log in again." if status == 401 else message
        expect(page.get_by_text(expected, exact=True)).to_be_visible()
        expect(page.locator('input[name="org"]')).to_have_value("Workshop_org")
        expect(page.locator('input[name="org"]')).to_be_enabled()
        if status == 401:
            # Authentication failures clear the session; retry requires a new login.
            return
        page.unroute("**/api/v1/users/*")
        page.get_by_role("button", name="Save", exact=True).click()
        expect(page.get_by_role("button", name="Add Site", exact=True)).to_be_visible()
    finally:
        with app.app_context():
            user = User.query.filter_by(email=email).first()
            if user:
                Store.delete_user(user.id)


def test_site_policy_rejection_preserves_editor_without_session_timeout(page, live_dashboard):
    app, _ = live_dashboard
    email = f"stale-site-{uuid4().hex}@test.com"
    with app.app_context():
        user = Store.create_user({"email": email, "password": "p", "role": "org_admin", "organization": "Stale_org"})[
            "user"
        ]
        site = Store.create_client({"name": "stale-" + uuid4().hex, "organization": "Stale_org"}, email)["client"]
    try:
        login(page, email, "p", 3)
        page.goto(NS + "/user-dashboard")
        expect(page.get_by_text(site["name"], exact=True)).to_be_visible()
        # The administrator approves the site while its owner still has the pending row open.
        with app.app_context():
            Store.patch_client_by_project_admin(site["id"], {"approval_state": 100})
        page.get_by_text(site["name"], exact=True).click()
        editor = page.locator('input[title="name"]')
        editor.fill("unapproved-name")
        validation_dialog(
            page, lambda: editor.press("Tab"), "Pending applicants can manage only their pending client sites."
        )
        expect(editor).to_have_value("unapproved-name")
        assert page.url.endswith("/user-dashboard")
        with app.app_context():
            assert db.session.get(Client, site["id"]).name == site["name"]
    finally:
        with app.app_context():
            Store.delete_user(user["id"])


def test_bulk_approval_keeps_all_successful_users_approved(page, live_dashboard):
    from contextlib import ExitStack

    app, _ = live_dashboard
    users = []
    try:
        with app.app_context():
            for _ in range(2):
                users.append(
                    Store.create_user(
                        {
                            "email": f"bulk-{uuid4().hex}@test.com",
                            "password": "p",
                            "organization": "Bulk_org",
                            "role": "org_admin",
                        }
                    )["user"]
                )
        login(page, TEST_USER, TEST_PW, 4)
        page.goto(NS + "/project-admin-dashboard")
        for user in users:
            row = page.get_by_role("row").filter(has_text=user["email"])
            expect(row).to_contain_text("Org Admin (requested)")
            row.get_by_role("checkbox").check()
        page.get_by_role("button", name="Bulk Actions", exact=True).click()
        page.get_by_role("button", name="Approve User", exact=True).click()
        with ExitStack() as stack:
            responses = [
                stack.enter_context(
                    page.expect_response(
                        lambda response, id=user["id"]: response.request.method == "PATCH"
                        and response.url.endswith(f"/users/{id}")
                    )
                )
                for user in users
            ]
            page.get_by_role("button", name="Apply", exact=True).click()
        for response in responses:
            assert response.value.status == 200
        with app.app_context():
            for user in users:
                assert db.session.get(User, user["id"]).approval_state == 100
        for user in users:
            row = page.get_by_role("row").filter(has_text=user["email"])
            expect(row).to_contain_text("Approved (D/L:")
            expect(row).not_to_contain_text("Org Admin (requested)")
    finally:
        with app.app_context():
            for user in users:
                Store.delete_user(user["id"])
