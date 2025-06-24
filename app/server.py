from flask import Flask
from app.routes import main  # or just `from routes import main` if you're inside app/

def create_app():
    try:
        app = Flask(__name__)
        app.config['UPLOAD_FOLDER'] = 'uploads'
        app.register_blueprint(main)
        print("✅ Blueprint registered and config set.")
        return app
    except Exception as e:
        print("❌ Exception in create_app():", e)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        print("🚀 Starting Flask app...")
        app = create_app()
        print("✅ Flask app created successfully.")
        app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        import traceback
        print("❌ Exception occurred while starting the Flask app:")
        traceback.print_exc()
