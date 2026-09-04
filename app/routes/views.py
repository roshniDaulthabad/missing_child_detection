from flask import Blueprint, render_template
from app.config import Config
from app.database.models import Camera, MissingChild, PotentialMatchAlert

views_bp = Blueprint("views", __name__)

@views_bp.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard", mode=Config.OPERATION_MODE)

@views_bp.route("/footage")
def footage():
    return render_template("footage.html", active_page="footage", mode=Config.OPERATION_MODE)

@views_bp.route("/live")
def live_stream():
    cameras = Camera.query.all()
    return render_template("live_stream.html", active_page="live", cameras=cameras, mode=Config.OPERATION_MODE)

@views_bp.route("/alerts")
def alerts():
    return render_template("alerts.html", active_page="alerts", mode=Config.OPERATION_MODE)

@views_bp.route("/cases")
def cases():
    return render_template("cases.html", active_page="cases", mode=Config.OPERATION_MODE)

@views_bp.route("/status")
def status_page():
    return render_template("status.html", active_page="status", mode=Config.OPERATION_MODE)

@views_bp.route("/movement")
def movement():
    cases = MissingChild.query.all()
    return render_template("movement.html", active_page="movement", cases=cases, mode=Config.OPERATION_MODE)

@views_bp.route("/welfare")
def welfare():
    cases = MissingChild.query.all()
    return render_template("welfare.html", active_page="welfare", cases=cases, mode=Config.OPERATION_MODE)

@views_bp.route("/timeline")
def timeline():
    cases = MissingChild.query.all()
    return render_template("timeline.html", active_page="timeline", cases=cases, mode=Config.OPERATION_MODE)

@views_bp.route("/settings")
def settings():
    return render_template("settings.html", active_page="settings", config=Config)
