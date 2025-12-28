"""
Sentinel-X Flask Application
Main application factory and configuration module
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, g
from werkzeug.exceptions import HTTPException


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(app):
    """
    Configure logging for the Flask application
    
    Sets up:
    - Console handler for stdout
    - Rotating file handler for persistent logs
    - Appropriate log levels and formatting
    """
    # Remove default Flask logger handlers
    app.logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Log format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    app.logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, 'sentinel-x.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    app.logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, 'sentinel-x-error.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    app.logger.addHandler(error_handler)
    
    # Set application logger level
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    app.logger.setLevel(getattr(logging, log_level))
    
    app.logger.info('Logging configured successfully')


# ============================================================================
# Error Handlers
# ============================================================================

def register_error_handlers(app):
    """
    Register error handlers for the Flask application
    Handles HTTP exceptions, validation errors, and general exceptions
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors"""
        app.logger.warning(f'Bad request: {error}')
        return jsonify({
            'status': 'error',
            'code': 400,
            'message': 'Bad Request: Invalid request parameters',
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle unauthorized errors"""
        app.logger.warning(f'Unauthorized access attempt: {error}')
        return jsonify({
            'status': 'error',
            'code': 401,
            'message': 'Unauthorized: Authentication required',
            'timestamp': datetime.utcnow().isoformat()
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle forbidden errors"""
        app.logger.warning(f'Forbidden access attempt: {error}')
        return jsonify({
            'status': 'error',
            'code': 403,
            'message': 'Forbidden: Access denied',
            'timestamp': datetime.utcnow().isoformat()
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors"""
        app.logger.debug(f'Resource not found: {request.path}')
        return jsonify({
            'status': 'error',
            'code': 404,
            'message': 'Not Found: Resource does not exist',
            'timestamp': datetime.utcnow().isoformat()
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle method not allowed errors"""
        app.logger.warning(f'Method not allowed: {request.method} {request.path}')
        return jsonify({
            'status': 'error',
            'code': 405,
            'message': 'Method Not Allowed',
            'timestamp': datetime.utcnow().isoformat()
        }), 405
    
    @app.errorhandler(422)
    def unprocessable_entity(error):
        """Handle validation errors"""
        app.logger.warning(f'Validation error: {error}')
        return jsonify({
            'status': 'error',
            'code': 422,
            'message': 'Unprocessable Entity: Validation failed',
            'timestamp': datetime.utcnow().isoformat()
        }), 422
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limiting errors"""
        app.logger.warning(f'Rate limit exceeded for {request.remote_addr}')
        return jsonify({
            'status': 'error',
            'code': 429,
            'message': 'Too Many Requests: Rate limit exceeded',
            'timestamp': datetime.utcnow().isoformat()
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors"""
        app.logger.error(f'Internal server error: {error}', exc_info=True)
        return jsonify({
            'status': 'error',
            'code': 500,
            'message': 'Internal Server Error: An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle service unavailable errors"""
        app.logger.error(f'Service unavailable: {error}')
        return jsonify({
            'status': 'error',
            'code': 503,
            'message': 'Service Unavailable: Please try again later',
            'timestamp': datetime.utcnow().isoformat()
        }), 503
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle all other HTTP exceptions"""
        app.logger.warning(f'HTTP exception: {error.code} - {error.description}')
        return jsonify({
            'status': 'error',
            'code': error.code,
            'message': error.description,
            'timestamp': datetime.utcnow().isoformat()
        }), error.code
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle unexpected exceptions"""
        app.logger.error(f'Unexpected exception: {str(error)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'code': 500,
            'message': 'Internal Server Error: An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


# ============================================================================
# Request/Response Handlers
# ============================================================================

def register_request_response_handlers(app):
    """
    Register request and response handlers for logging and processing
    """
    
    @app.before_request
    def before_request():
        """Execute before each request"""
        # Store request start time
        g.request_start_time = datetime.utcnow()
        
        # Log incoming request
        app.logger.debug(
            f'{request.method} {request.path} - '
            f'Remote: {request.remote_addr} - '
            f'User-Agent: {request.headers.get("User-Agent", "Unknown")}'
        )
        
        # Store request info in g object
        g.request_id = request.headers.get('X-Request-ID', None)
        g.method = request.method
        g.path = request.path
        g.remote_addr = request.remote_addr
    
    @app.after_request
    def after_request(response):
        """Execute after each request"""
        # Calculate request duration
        if hasattr(g, 'request_start_time'):
            duration = (datetime.utcnow() - g.request_start_time).total_seconds()
        else:
            duration = 0
        
        # Log response
        app.logger.debug(
            f'{request.method} {request.path} - '
            f'Status: {response.status_code} - '
            f'Duration: {duration:.3f}s'
        )
        
        # Add common response headers
        response.headers['X-Request-Duration'] = str(duration)
        response.headers['X-Powered-By'] = 'Sentinel-X'
        
        if hasattr(g, 'request_id') and g.request_id:
            response.headers['X-Request-ID'] = g.request_id
        
        return response
    
    @app.teardown_request
    def teardown_request(exception):
        """Execute at end of request lifecycle"""
        if exception:
            app.logger.error(f'Request teardown exception: {str(exception)}', exc_info=True)


# ============================================================================
# CLI Commands
# ============================================================================

def register_cli_commands(app):
    """
    Register custom CLI commands for the Flask application
    """
    
    @app.cli.command('init-db')
    def init_db():
        """Initialize the database"""
        app.logger.info('Initializing database...')
        try:
            # Import and call database initialization
            from backend.db import init_database
            init_database(app)
            print('✓ Database initialized successfully')
            app.logger.info('Database initialization completed')
        except Exception as e:
            app.logger.error(f'Database initialization failed: {str(e)}')
            print(f'✗ Database initialization failed: {str(e)}')
    
    @app.cli.command('seed-db')
    def seed_db():
        """Seed the database with initial data"""
        app.logger.info('Seeding database...')
        try:
            # Import and call database seeding
            from backend.db import seed_database
            seed_database(app)
            print('✓ Database seeded successfully')
            app.logger.info('Database seeding completed')
        except Exception as e:
            app.logger.error(f'Database seeding failed: {str(e)}')
            print(f'✗ Database seeding failed: {str(e)}')
    
    @app.cli.command('create-admin')
    def create_admin():
        """Create an admin user"""
        app.logger.info('Creating admin user...')
        try:
            # Import user creation function
            from backend.auth import create_admin_user
            admin = create_admin_user(app)
            print(f'✓ Admin user created successfully')
            print(f'  Email: {admin.get("email")}')
            app.logger.info(f'Admin user created: {admin.get("email")}')
        except Exception as e:
            app.logger.error(f'Admin user creation failed: {str(e)}')
            print(f'✗ Admin user creation failed: {str(e)}')
    
    @app.cli.command('run-tests')
    def run_tests():
        """Run application tests"""
        app.logger.info('Running tests...')
        try:
            import subprocess
            result = subprocess.run(
                ['pytest', 'tests/', '-v', '--cov=backend'],
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            app.logger.info(f'Tests completed with return code: {result.returncode}')
        except Exception as e:
            app.logger.error(f'Test execution failed: {str(e)}')
            print(f'✗ Test execution failed: {str(e)}')
    
    @app.cli.command('show-config')
    def show_config():
        """Display current application configuration"""
        print('Application Configuration:')
        print('-' * 50)
        config_items = [
            'DEBUG',
            'TESTING',
            'ENVIRONMENT',
            'DATABASE_URL',
            'SECRET_KEY',
            'JSON_SORT_KEYS'
        ]
        for key in config_items:
            value = app.config.get(key, 'Not Set')
            if 'SECRET' in key or 'PASSWORD' in key:
                value = '***' if value and value != 'Not Set' else value
            print(f'{key}: {value}')
        app.logger.info('Configuration displayed')


# ============================================================================
# Application Factory
# ============================================================================

def create_app(config_name=None):
    """
    Application factory function
    
    Creates and configures a Flask application instance
    
    Args:
        config_name (str, optional): Configuration environment name.
                                    Defaults to environment variable FLASK_ENV or 'development'
    
    Returns:
        Flask: Configured Flask application instance
    """
    
    # Determine configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development').lower()
    
    # Create Flask application
    app = Flask(__name__)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    # Base configuration
    app.config['ENVIRONMENT'] = config_name
    app.config['JSON_SORT_KEYS'] = False
    
    # Secret key
    secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SECRET_KEY'] = secret_key
    
    # Database configuration
    database_url = os.getenv(
        'DATABASE_URL',
        'sqlite:///sentinel_x.db' if config_name == 'development' else None
    )
    if database_url:
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Environment-specific configuration
    if config_name == 'development':
        app.config['DEBUG'] = True
        app.config['TESTING'] = False
        app.config['PROPAGATE_EXCEPTIONS'] = True
    elif config_name == 'testing':
        app.config['DEBUG'] = False
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    elif config_name == 'production':
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    # Setup logging
    setup_logging(app)
    app.logger.info(f'Creating Flask application in {config_name} mode')
    
    # Initialize extensions (if any exist)
    try:
        from backend.extensions import init_extensions
        init_extensions(app)
        app.logger.info('Extensions initialized')
    except ImportError:
        app.logger.debug('No extensions module found, skipping initialization')
    except Exception as e:
        app.logger.warning(f'Extension initialization warning: {str(e)}')
    
    # ========================================================================
    # Request/Response Handlers
    # ========================================================================
    
    register_request_response_handlers(app)
    
    # ========================================================================
    # Error Handlers
    # ========================================================================
    
    register_error_handlers(app)
    
    # ========================================================================
    # Blueprint Registration
    # ========================================================================
    
    register_blueprints(app)
    
    # ========================================================================
    # CLI Commands
    # ========================================================================
    
    register_cli_commands(app)
    
    # ========================================================================
    # Application Context
    # ========================================================================
    
    @app.shell_context_processor
    def make_shell_context():
        """Make shell context for flask shell command"""
        return {}
    
    # ========================================================================
    # Health Check Route
    # ========================================================================
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'environment': app.config['ENVIRONMENT']
        }), 200
    
    @app.route('/api/v1/health', methods=['GET'])
    def api_health_check():
        """API health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'Sentinel-X',
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'environment': app.config['ENVIRONMENT']
        }), 200
    
    app.logger.info(f'Flask application created successfully in {config_name} mode')
    
    return app


# ============================================================================
# Blueprint Registration
# ============================================================================

def register_blueprints(app):
    """
    Register all application blueprints
    
    Blueprints should be located in the backend/routes directory
    Each blueprint module should export a 'bp' variable
    """
    
    blueprints_dir = os.path.join(os.path.dirname(__file__), 'routes')
    
    # List of blueprint modules to register
    blueprint_modules = [
        # 'auth',      # Authentication routes
        # 'api',       # API routes
        # 'admin',     # Admin routes
        # 'public',    # Public routes
    ]
    
    for module_name in blueprint_modules:
        try:
            module = __import__(f'backend.routes.{module_name}', fromlist=['bp'])
            if hasattr(module, 'bp'):
                app.register_blueprint(module.bp)
                app.logger.info(f'Blueprint registered: {module_name}')
            else:
                app.logger.warning(f'Blueprint module {module_name} has no "bp" attribute')
        except ImportError as e:
            app.logger.debug(f'Blueprint not found or not yet created: {module_name}')
        except Exception as e:
            app.logger.error(f'Error registering blueprint {module_name}: {str(e)}')


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=app.config.get('DEBUG', False)
    )
