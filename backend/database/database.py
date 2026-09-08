"""
Database engine, session, and base model setup.

Uses SQLite for zero-setup portability (Contract §26).
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config import get_config

_config = get_config()

engine = create_engine(
    _config.server.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def init_db() -> None:
    """Create tables and apply additive compatibility migrations.

    ``create_all`` intentionally does not alter tables that already exist.  The
    prototype database may therefore pre-date fields added to the persisted
    impact and recovery contracts.  Keep those databases usable by adding only
    the missing nullable/defaulted columns at startup.
    """
    import backend.database.models  # Ensure models are registered
    Base.metadata.create_all(bind=engine)

    if engine.dialect.name != "sqlite":
        return

    additive_columns = {
        "impact_records": {
            "impact_breakdown_json": "TEXT DEFAULT '{}'",
            "top_impacted_layers_json": "TEXT DEFAULT '[]'",
        },
        "recovery_records": {
            "selected_action": "VARCHAR",
            "details_json": "TEXT DEFAULT '{}'",
        },
    }

    schema = inspect(engine)
    existing_tables = set(schema.get_table_names())
    with engine.begin() as connection:
        for table_name, expected_columns in additive_columns.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {
                column["name"] for column in schema.get_columns(table_name)
            }
            for column_name, column_definition in expected_columns.items():
                if column_name not in existing_columns:
                    connection.exec_driver_sql(
                        f'ALTER TABLE "{table_name}" '
                        f'ADD COLUMN "{column_name}" {column_definition}'
                    )


def get_db():
    """Yield a database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
