from flask import Flask
import os

def create_app():
    """
    Factory function to create and configure the Flask app.

    Returns:
    - app (Flask): Configured Flask application.
    """
    app = Flask(__name__)

    # Set configuration options (if any)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')

    # Import routes after creating the app to avoid circular imports
    from . import routes

    # Register routes with the app
    app.register_blueprint(routes.bp)

    return app
