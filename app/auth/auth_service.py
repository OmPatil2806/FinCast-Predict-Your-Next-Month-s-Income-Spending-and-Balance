"""Registration and login logic. Passwords are only ever stored as bcrypt hashes."""

import re
import sqlite3
from pathlib import Path

import bcrypt

from app.database.crud import (
    create_user, get_user_by_email, get_user_by_id, get_user_by_username, update_password_hash,
)

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt only accepts up to 72 bytes

LOGIN_FAILED = "Invalid username/email or password."


class AuthError(Exception):
    """Raised with a user-friendly message when registration fails."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthError("Password is too long (maximum 72 bytes).")


def validate_registration(username: str, email: str, password: str) -> None:
    """Raise AuthError with a clear message if any field is invalid."""
    if not USERNAME_PATTERN.match(username.strip()):
        raise AuthError("Username must be 3-30 characters: letters, numbers or _.")
    if not EMAIL_PATTERN.match(email.strip()):
        raise AuthError("Please enter a valid email address.")
    validate_password(password)


def register(username: str, email: str, password: str,
             db_path: str | Path | None = None) -> int:
    """Validate the input, hash the password and create the user. Return the new user id."""
    validate_registration(username, email, password)
    if get_user_by_username(username, db_path):
        raise AuthError("This username is already taken.")
    if get_user_by_email(email, db_path):
        raise AuthError("An account with this email already exists.")
    try:
        return create_user(username, email, hash_password(password), db_path=db_path)
    except sqlite3.IntegrityError:
        # someone registered the same name between our check and the insert
        raise AuthError("This username or email is already taken.")


def login(username_or_email: str, password: str,
          db_path: str | Path | None = None) -> dict | None:
    """Return the user (without the password hash) if the credentials are correct, else None.

    The same None is returned for an unknown user and a wrong password,
    so the login form never reveals which usernames exist.
    """
    identifier = username_or_email.strip()
    if not identifier or not password:
        return None
    if "@" in identifier:
        user = get_user_by_email(identifier, db_path)
    else:
        user = get_user_by_username(identifier, db_path)
    if user is None or not verify_password(password, user["password_hash"]):
        return None
    user.pop("password_hash")
    return user


def change_password(user_id: int, current_password: str, new_password: str,
                    db_path: str | Path | None = None) -> None:
    """Check the current password, validate the new one and store its hash. Raises AuthError."""
    user = get_user_by_id(user_id, db_path)
    if user is None or not verify_password(current_password or "", user["password_hash"]):
        raise AuthError("Your current password is not correct.")
    validate_password(new_password)
    if verify_password(new_password, user["password_hash"]):
        raise AuthError("The new password must be different from the current one.")
    update_password_hash(user_id, hash_password(new_password), db_path)
