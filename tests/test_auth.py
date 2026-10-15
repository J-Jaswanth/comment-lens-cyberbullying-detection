from werkzeug.security import check_password_hash

from app.routes.auth import _safe_next_url


def register(client, username="student_1", email="student@example.com", password="strong-password-123"):
    return client.post(
        "/auth/register",
        data={
            "username": username,
            "email": email,
            "password": password,
            "confirm_password": password,
        },
    )


def test_registration_stores_a_password_hash(client, account_store):
    response = register(client)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/auth/login")
    user = account_store.get_by_login("student_1")
    assert user is not None
    assert user.password_hash != "strong-password-123"
    assert check_password_hash(user.password_hash, "strong-password-123")


def test_registration_rejects_short_password(client, account_store):
    response = register(client, password="short123")

    assert response.status_code == 200
    assert not account_store.users
    assert b"at least 12 characters" in response.data


def test_login_logout_and_email_identifier(client):
    register(client)
    response = client.post(
        "/auth/login",
        data={"identifier": "STUDENT@example.com", "password": "strong-password-123"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    assert b"Welcome, student_1" in client.get("/").data

    logout_response = client.post("/auth/logout")
    assert logout_response.status_code == 302
    assert client.get("/").status_code == 302


def test_login_rejects_invalid_password(client):
    register(client)
    response = client.post(
        "/auth/login",
        data={"identifier": "student_1", "password": "incorrect-password"},
    )

    assert response.status_code == 200
    assert b"incorrect" in response.data
    assert client.get("/").status_code == 302


def test_login_rejects_external_next_url(client):
    register(client)
    response = client.post(
        "/auth/login?next=https://attacker.example/",
        data={"identifier": "student_1", "password": "strong-password-123"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_safe_next_url_rejects_protocol_relative_and_backslash_paths():
    assert _safe_next_url("//attacker.example/") is None
    assert _safe_next_url("/\\attacker.example/") is None


def test_registration_requires_csrf_token(csrf_client):
    response = csrf_client.post("/auth/register", data={})

    assert response.status_code == 400