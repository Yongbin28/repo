import pytest
from waferpulse.core.auth import load_user_database, verify_credentials, _hash_password


def test_auth_verify_valid_credentials():
    admin = verify_credentials("admin", "admin123")
    assert admin is not None
    assert admin["role"] == "Lead Administrator"

    eng = verify_credentials("engineer", "engineer123")
    assert eng is not None
    assert eng["role"] == "Process Engineer"


def test_auth_verify_invalid_password():
    res = verify_credentials("admin", "wrongpassword")
    assert res is None


def test_auth_verify_unknown_user():
    res = verify_credentials("nonexistent_user", "password")
    assert res is None
