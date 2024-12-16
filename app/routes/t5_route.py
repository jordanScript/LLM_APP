from flask import Blueprint, request, jsonify
from app.services.t5_service import query

bp = Blueprint("t5", __name__, url_prefix="/t5")

@bp.route("/flan-t5", methods=["POST"])
def flan_t5_promt():
    data = request.json 

    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    prompt = f"""{data.get('prompt')}"""
    response = query(prompt)

    return jsonify({"response": response})