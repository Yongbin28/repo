#!/usr/bin/env python3
"""
WaferPulse User Management CLI.
Utility to list, add, update, and remove user accounts for cloud authentication.

Usage:
    python manage_users.py list
    python manage_users.py add <username> <full_name> <role> <password>
    python manage_users.py passwd <username> <new_password>
    python manage_users.py delete <username>
"""

import sys
import json
import hashlib
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config" / "users.json"


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_db() -> dict:
    if not CONFIG_FILE.exists():
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        return {"users": {}}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {CONFIG_FILE}: {e}")
        return {"users": {}}


def _save_db(db: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"✓ Configuration saved to {CONFIG_FILE.name}")


def list_users() -> None:
    db = _load_db()
    users = db.get("users", {})
    print("\n== WaferPulse Registered Users ==")
    if not users:
        print("  (No users found)")
        return
    print(f"  {'Username':<15} {'Role':<25} {'Full Name'}")
    print("  " + "-" * 60)
    for u, data in users.items():
        role = data.get("role", "User")
        name = data.get("name", u)
        print(f"  {u:<15} {role:<25} {name}")
    print()


def add_user(username: str, full_name: str, role: str, password: str) -> None:
    db = _load_db()
    u = username.strip().lower()
    if u in db.get("users", {}):
        print(f"Error: User '{u}' already exists. Use 'passwd' to update password.")
        sys.exit(1)
    if "users" not in db:
        db["users"] = {}
    db["users"][u] = {
        "name": full_name.strip(),
        "password_hash": _hash_password(password),
        "role": role.strip(),
    }
    _save_db(db)
    print(f"✓ Added user '{u}' ({full_name}) with role '{role}'.")


def set_password(username: str, new_password: str) -> None:
    db = _load_db()
    u = username.strip().lower()
    if u not in db.get("users", {}):
        print(f"Error: User '{u}' not found.")
        sys.exit(1)
    db["users"][u]["password_hash"] = _hash_password(new_password)
    _save_db(db)
    print(f"✓ Password updated successfully for user '{u}'.")


def delete_user(username: str) -> None:
    db = _load_db()
    u = username.strip().lower()
    if u not in db.get("users", {}):
        print(f"Error: User '{u}' not found.")
        sys.exit(1)
    del db["users"][u]
    _save_db(db)
    print(f"✓ Deleted user '{u}'.")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        list_users()
        return

    cmd = args[0].lower()
    if cmd == "list":
        list_users()
    elif cmd == "add":
        if len(args) < 5:
            print("Usage: python manage_users.py add <username> <full_name> <role> <password>")
            sys.exit(1)
        add_user(args[1], args[2], args[3], args[4])
    elif cmd in ("passwd", "password"):
        if len(args) < 3:
            print("Usage: python manage_users.py passwd <username> <new_password>")
            sys.exit(1)
        set_password(args[1], args[2])
    elif cmd in ("delete", "remove", "rm"):
        if len(args) < 2:
            print("Usage: python manage_users.py delete <username>")
            sys.exit(1)
        delete_user(args[1])
    else:
        print(f"Unknown command: '{cmd}'. Run with --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
