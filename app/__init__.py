from flask import Flask
import logging

def create_app():
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)
    app.config.from_object('app.config.Config')
    from .routes import main
    app.register_blueprint(main)
    return app
