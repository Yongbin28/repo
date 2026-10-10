"""
Authentication module for WaferPulse.
Provides secure sign-in, session state management, and role-based access control.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

AUTH_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "users.json"


def _hash_password(password: str) -> str:
    """Return SHA-256 hash of plaintext password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_user_database() -> Dict[str, Any]:
    """Load user credentials from config/users.json."""
    if not AUTH_FILE.exists():
        AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        default_db = {
            "users": {
                "admin": {
                    "name": "Quality Lead Administrator",
                    "password_hash": _hash_password("admin123"),
                    "role": "Lead Administrator",
                },
                "engineer": {
                    "name": "Fab Process Engineer",
                    "password_hash": _hash_password("engineer123"),
                    "role": "Process Engineer",
                },
                "auditor": {
                    "name": "Quality Auditor",
                    "password_hash": _hash_password("auditor123"),
                    "role": "Quality Auditor",
                },
            }
        }
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(default_db, f, indent=2)
        return default_db

    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}}


def verify_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify username and password. Returns user dict if valid, None otherwise."""
    db = load_user_database()
    users = db.get("users", {})
    user_info = users.get(username.strip().lower())
    if not user_info:
        return None
    if user_info.get("password_hash") == _hash_password(password):
        return user_info
    return None


def render_auth_gate() -> bool:
    """
    Renders login screen if not authenticated.
    Returns True if user is logged in, False otherwise.
    """
    # Allow environment variable override if auth is disabled in dev mode
    if os.environ.get("WAFERPULSE_AUTH_DISABLED", "false").lower() == "true":
        return True

    if st.session_state.get("authenticated", False):
        # Render User Badge in Sidebar
        user_name = st.session_state.get("auth_user_name", "User")
        user_role = st.session_state.get("auth_user_role", "Engineer")
        username = st.session_state.get("auth_username", "")

        st.sidebar.markdown(
            f"""
            <div style="background: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
                <div style="font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase;">Authenticated User</div>
                <div style="font-size: 14px; font-weight: 700; color: #0F172A;">{user_name}</div>
                <div style="font-size: 12px; color: #2563EB;">{user_role} ({username})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.sidebar.button("🚪 Sign Out", width="stretch", type="secondary"):
            st.session_state["authenticated"] = False
            st.session_state["auth_username"] = ""
            st.session_state["auth_user_name"] = ""
            st.session_state["auth_user_role"] = ""
            st.rerun()

        return True

    # Render Centered Login Card
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 40px; margin-bottom: 25px;">
                <h1 style="font-size: 32px; font-weight: 800; color: #0F172A; margin-bottom: 6px;">
                    🛡️ WaferPulse™
                </h1>
                <p style="font-size: 14px; color: #64748B; margin-top: 0;">
                    Semiconductor Explainable AI Quality Risk Platform
                </p>
                <div style="display: inline-block; background: #EEF2FF; border: 1px solid #C7D2FE; color: #4338CA; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                    Cloud Access Sign-In
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### Please Sign In to Access Platform")
            username_input = st.text_input("Username", placeholder="e.g. engineer, admin", key="input_user")
            password_input = st.text_input("Password", type="password", placeholder="Enter your password", key="input_pwd")
            submit_login = st.form_submit_button("Sign In to WaferPulse", width="stretch", type="primary")

            if submit_login:
                if not username_input or not password_input:
                    st.error("Please enter both username and password.")
                else:
                    user_record = verify_credentials(username_input, password_input)
                    if user_record:
                        st.session_state["authenticated"] = True
                        st.session_state["auth_username"] = username_input.strip().lower()
                        st.session_state["auth_user_name"] = user_record.get("name", username_input)
                        st.session_state["auth_user_role"] = user_record.get("role", "Engineer")
                        st.success(f"Welcome, {user_record.get('name')}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")

    return False
