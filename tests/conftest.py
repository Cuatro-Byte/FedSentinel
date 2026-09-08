"""
Shared test configuration and fixtures.
"""

import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.database import Base
from backend.adapters.registry import configure_adapters, reset_adapters
from tests.fixtures.mock_p1 import MockP1FL
from tests.fixtures.mock_p2 import MockP2Attack
from tests.fixtures.mock_p3 import MockP3Sentinel


from sqlalchemy.pool import StaticPool

@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_p1():
    """Provide a fresh MockP1FL instance."""
    return MockP1FL()


@pytest.fixture
def mock_p2():
    """Provide a fresh MockP2Attack instance."""
    return MockP2Attack()


@pytest.fixture
def mock_p3():
    """Provide a fresh MockP3Sentinel instance."""
    return MockP3Sentinel()


import unittest.mock as mock

@pytest.fixture(autouse=True)
def configured_adapters():
    """Explicitly inject test doubles; production never imports test fixtures."""
    configure_adapters(MockP1FL(), MockP2Attack(), MockP3Sentinel())
    with mock.patch("backend.main.configure_adapters"):
        yield
    reset_adapters()
