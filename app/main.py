from flask import Flask
from app.routes import t5_route

def create_app():
    app = Flask(__name__)

    # Registra blueprints
    app.register_blueprint(t5_route.bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(port=5000, debug=True)