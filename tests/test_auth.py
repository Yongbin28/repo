import pytest
from waferpulse.core.auth import load_user_database, verify_credentials, _hash_password


def test_auth_verify_valid_credentials():
    admin = verify_credentials("admin", "admin321")
    assert admin is not None
    assert admin["role"] == "Lead Administrator"

    # Also test role321 alias
    admin_alt = verify_credentials("admin", "role321")
    assert admin_alt is not None

    eng = verify_credentials("engineer", "engineer321")
    assert eng is not None
    assert eng["role"] == "Process Engineer"

    eng_alt = verify_credentials("engineer", "role321")
    assert eng_alt is not None


def test_auth_verify_invalid_password():
    res = verify_credentials("admin", "wrongpassword")
    assert res is None


def test_auth_verify_unknown_user():
    res = verify_credentials("nonexistent_user", "password")
    assert res is None
