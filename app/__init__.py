import os
from flask import Flask
from app.config import Config
from app.database.models import db, User, Camera
from app.routes.api import api_bp
from app.routes.views import views_bp
from app.services.case_service import case_service

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize SQLAlchemy
    db.init_app(app)

    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)

    with app.app_context():
        # Create database tables
        db.create_all()

        # Seed initial authorized cameras if empty
        if Camera.query.count() == 0:
            default_cameras = [
                Camera(camera_id="CAM-01", name="Main Gate CCTV", location="Sector 4 Main Gate", latitude=28.6139, longitude=77.2090, fps=15.0, status="online"),
                Camera(camera_id="CAM-02", name="Bus Terminal Transit", location="Interstate Bus Stand", latitude=28.6210, longitude=77.2150, fps=15.0, status="online"),
                Camera(camera_id="CAM-03", name="Railway Station West", location="Platform 1 Exit", latitude=28.6320, longitude=77.2200, fps=15.0, status="online"),
                Camera(camera_id="CAM-04", name="City Market South", location="Central Bazaar Entry", latitude=28.6280, longitude=77.2110, fps=15.0, status="online"),
            ]
            db.session.add_all(default_cameras)
            db.session.commit()

        # Seed default officer if empty
        if User.query.count() == 0:
            officer = User(username="officer", full_name="Inspector R. Sharma", role="investigator", badge_number="DL-8821")
            officer.set_password("police123")
            db.session.add(officer)
            db.session.commit()

        # Load active embeddings from database into memory cache
        case_service.load_active_embeddings_to_cache()

    return app

# Singleton app instance
app = create_app()
