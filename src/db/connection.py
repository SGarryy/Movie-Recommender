import os
import urllib
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUE_VALUES


def _build_odbc_connect_string() -> str:
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    encrypt = os.getenv("DB_ENCRYPT", "no")
    trust_server_certificate = "yes" if _env_flag("DB_TRUST_SERVER_CERTIFICATE") else "no"

    if not server or not database:
        raise ValueError(
            "DB_SERVER and DB_NAME must be set before creating a database connection."
        )

    params = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Encrypt={encrypt}",
        f"TrustServerCertificate={trust_server_certificate}",
    ]

    if user:
        params.extend([f"UID={user}", f"PWD={password or ''}"])
    else:
        params.append("Trusted_Connection=yes")

    return ";".join(params) + ";"


@lru_cache(maxsize=1)
def get_connection() -> Engine:
    params = urllib.parse.quote_plus(_build_odbc_connect_string())
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        future=True,
        pool_pre_ping=True,
        hide_parameters=True,
    )


def validate_connection() -> None:
    with get_connection().connect() as connection:
        connection.execute(text("SELECT 1"))


def clear_connection_cache() -> None:
    get_connection.cache_clear()
