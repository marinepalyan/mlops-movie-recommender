import logging
import os
import pathlib
import sys

import api

import typing as t
from pydantic import BaseModel, validator
from yaml import safe_load

# logging format
FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)

# Project Directories
ROOT = pathlib.Path(api.__file__).resolve().parent.parent

CONFIG_FILE_PATH = ROOT / "config.yml"
TRAINED_MODEL_DIR = ROOT / "models"
DATASET_DIR = ROOT / "data"

class Config(BaseModel):
    DEBUG = False
    TESTING = False
    ENV = os.getenv("FLASK_ENV", "production")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", logging.INFO)
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:"  # dialect = postgresql, driver = psycopg2
        f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    )  # more on the URI here: https://docs.sqlalchemy.org/en/20/core/engines.html

    # DB config matches docker container
    DB_USER = os.getenv("DB_USER", "davit")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "19dv009")
    DB_PORT = os.getenv("DB_PORT", 7619)
    DB_HOST = os.getenv("DB_HOST", "0.0.0.0")
    DB_NAME = os.getenv("DB_NAME", "ml_api_dev")
    
    app_config: AppConfig
    model_config: ModelConfig


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"  # do not use in production!
    LOGGING_LEVEL = logging.DEBUG


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    LOGGING_LEVEL = logging.DEBUG

    # DB config matches docker container
    DB_USER = os.getenv("DB_USER", "test_davit")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "19dv009")
    DB_PORT = os.getenv("DB_PORT", 7618)
    DB_HOST = os.getenv("DB_HOST", "0.0.0.0")
    DB_NAME = "ml_api_test"
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:"
        f"{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    cleaned_data: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    drop_features: list[str]
    target: str
    test_size: float

        
class ProductionConfig(Config):
    pass


def get_console_handler():
    """Setup console logging handler."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def setup_app_logging(config: Config) -> None:
    """Prepare custom logging for our application."""
    _disable_irrelevant_loggers()
    root = logging.getLogger()
    root.setLevel(config.LOGGING_LEVEL)
    root.addHandler(get_console_handler())
    root.propagate = False


def _disable_irrelevant_loggers() -> None:
    """Disable loggers created by packages which create a lot of noise."""
    for logger_name in (
        "connexion.apis.flask_api",
        "connexion.apis.abstract",
        "connexion.decorators",
        "connexion.operation",
        "connexion.operations",
        "connexion.app",
        "openapi_spec_validator",
    ):
        logging.getLogger(logger_name).level = logging.WARNING

        
def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> t.Dict:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = safe_load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: t.Optional[t.Dict] = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_config=ModelConfig(**parsed_config),
    )

    return _config


config = create_and_validate_config()
