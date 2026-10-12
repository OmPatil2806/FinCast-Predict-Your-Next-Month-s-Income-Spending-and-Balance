import pytest

from app.auth.auth_service import AuthError, hash_password, login, register, verify_password
from app.database.crud import get_user_by_username
from app.database.db import get_connection, init_db

PASSWORD = "secret123"


@pytest.fixture
def db(tmp_path):
    """A fresh, empty database for every test - never the real data/app.db."""
    return init_db(tmp_path / "test.db")


@pytest.fixture
def user_id(db):
    return register("om_patil", "om@example.com", PASSWORD, db_path=db)


def test_hash_is_not_plain_password():
    hashed = hash_password(PASSWORD)
    assert hashed != PASSWORD
    assert hashed.startswith("$2b$")
    assert verify_password(PASSWORD, hashed)
    assert not verify_password("wrong-pass", hashed)


def test_register_returns_id_and_stores_only_hash(db, user_id):
    assert user_id == 1
    stored = get_user_by_username("om_patil", db)
    assert stored["password_hash"] != PASSWORD
    assert verify_password(PASSWORD, stored["password_hash"])


@pytest.mark.parametrize("username", ["om_patil", "OM_PATIL"])
def test_duplicate_username_rejected(db, user_id, username):
    with pytest.raises(AuthError, match="username is already taken"):
        register(username, "other@example.com", PASSWORD, db_path=db)


def test_duplicate_email_rejected(db, user_id):
    with pytest.raises(AuthError, match="email already exists"):
        register("someone_else", "OM@Example.com", PASSWORD, db_path=db)


@pytest.mark.parametrize("username, email, password, message", [
    ("ab", "a@b.co", PASSWORD, "Username"),
    ("om patil", "a@b.co", PASSWORD, "Username"),
    ("new_user", "not-an-email", PASSWORD, "valid email"),
    ("new_user", "a@b.co", "short", "at least 8"),
    ("new_user", "a@b.co", "x" * 73, "too long"),
])
def test_invalid_input_rejected(db, username, email, password, message):
    with pytest.raises(AuthError, match=message):
        register(username, email, password, db_path=db)


def test_login_with_username(db, user_id):
    user = login("om_patil", PASSWORD, db_path=db)
    assert user["id"] == user_id
    assert "password_hash" not in user


def test_login_with_email_any_case(db, user_id):
    user = login("OM@example.com", PASSWORD, db_path=db)
    assert user["username"] == "om_patil"


def test_wrong_password_rejected(db, user_id):
    assert login("om_patil", "wrong-pass", db_path=db) is None


def test_unknown_user_rejected(db):
    assert login("nobody", PASSWORD, db_path=db) is None


def test_empty_login_rejected(db, user_id):
    assert login("", "", db_path=db) is None


def test_delete_user_removes_profile(db, user_id):
    conn = get_connection(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM profiles WHERE user_id = ?", (user_id,)).fetchone()[0] == 1
        with conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        assert conn.execute("SELECT COUNT(*) FROM profiles WHERE user_id = ?", (user_id,)).fetchone()[0] == 0
    finally:
        conn.close()
