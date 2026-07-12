from flask import Flask, request, jsonify, Blueprint
# from convertion import convert_model as app
from convertion import convert_model

app = Flask(__name__)

app.register_blueprint(convert_model)


# 🔹 Health check route to confirm server is running
@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "message": "Conversion server is running"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
