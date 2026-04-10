import os
from pathlib import Path

from sqlmodel import SQLModel, create_engine, Session

# Default DB location is repo-root `app_data/database.db` (writable, and easy to reset).
# Override with QA_DB_PATH, e.g. QA_DB_PATH=/abs/path/to/database.db
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "app_data" / "database.db"
db_path = Path(os.getenv("QA_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()
db_path.parent.mkdir(parents=True, exist_ok=True)

sqlite_url = f"sqlite:///{db_path}"

engine = create_engine(sqlite_url, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
