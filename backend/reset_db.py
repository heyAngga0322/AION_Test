from pathlib import Path

from database import create_db_and_tables, db_path, engine
from sqlmodel import SQLModel


def reset_database() -> None:
    # Drop tables first to release SQLite schema cleanly.
    SQLModel.metadata.drop_all(engine)

    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()

    create_db_and_tables()
    print(f"Database reset complete: {db_file}")


if __name__ == "__main__":
    reset_database()
