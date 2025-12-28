"""
Configuration settings for Sentinel-X backend application.

This module contains all configuration variables and settings used throughout
the backend application, including database, API, logging, and environment-specific settings.
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class with common settings."""
    
    # Application Settings
    APP_NAME = "Sentinel-X"
    APP_VERSION = "1.0.0"
    DEBUG = False
    TESTING = False
    
    # Security Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-secret-key-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # CORS Settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    
    # Database Settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///sentinel.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # Redis Settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TIMEOUT = 300  # 5 minutes
    
    # API Settings
    API_TITLE = "Sentinel-X API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "RESTful API for Sentinel-X monitoring system"
    API_DOCS_URL = "/docs"
    API_OPENAPI_URL = "/openapi.json"
    
    # Logging Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "logs/sentinel-x.log"
    LOG_MAX_BYTES = 10485760  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Upload Settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_UPLOAD_SIZE = 104857600  # 100MB
    ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif", "csv", "json"}
    
    # Pagination Settings
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
    
    # Celery Settings (for async tasks)
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    
    # Email Settings
    MAIL_SERVER = os.getenv("MAIL_SERVER", "localhost")
    MAIL_PORT = int(os.getenv("MAIL_PORT", 587))
    MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", True)
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", "noreply@sentinel-x.local")
    
    # Third-party API Keys
    EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")
    EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "")
    
    # Feature Flags
    ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "True").lower() == "true"
    ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "True").lower() == "true"
    ENABLE_EXPORT = os.getenv("ENABLE_EXPORT", "True").lower() == "true"
    
    # Performance Settings
    WORKER_THREADS = int(os.getenv("WORKER_THREADS", 4))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
    
    # Data Retention Settings
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", 90))
    ARCHIVE_AFTER_DAYS = int(os.getenv("ARCHIVE_AFTER_DAYS", 30))


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    DATABASE_URL = os.getenv("DEV_DATABASE_URL", "sqlite:///sentinel_dev.db")
    LOG_LEVEL = "DEBUG"
    SQLALCHEMY_ECHO = True
    MAIL_USE_TLS = False
    MAIL_PORT = 1025  # MailHog default port


class TestingConfig(Config):
    """Testing configuration."""
    
    DEBUG = True
    TESTING = True
    DATABASE_URL = "sqlite:///:memory:"
    REDIS_URL = "redis://localhost:6379/15"
    CELERY_BROKER_URL = "memory://"
    CELERY_RESULT_BACKEND = "cache+memory://"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)
    RATELIMIT_ENABLED = False
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Ensure critical settings are provided in production
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not all([SECRET_KEY, JWT_SECRET_KEY, DATABASE_URL]):
        raise ValueError("Critical environment variables not set for production!")
    
    LOG_LEVEL = "INFO"
    SQLALCHEMY_ECHO = False


# Configuration mapping
config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def get_config(config_name=None):
    """
    Get configuration object based on environment.
    
    Args:
        config_name (str, optional): Name of configuration to load.
                                    If not provided, uses FLASK_ENV or defaults to 'development'.
    
    Returns:
        Config: Configuration class instance.
    """
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")
    
    return config_by_name.get(config_name, DevelopmentConfig)
