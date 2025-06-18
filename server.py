import io
import base64
import zipfile
import sqlite3
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from object_detection import ObjectDetection
from obj_classification import ObjClassification

app = Flask(__name__)
CORS(app)

@app.route('/detect', methods = ['POST'])
def DetectObjects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    fragments = ObjectDetection(file.stream)
    if not fragments:
        return jsonify({'error': 'No objects detected'}), 400
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, buffer in fragments:
            zipf.writestr(filename, buffer.getvalue())

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name="fragments.zip")

@app.route('/classify', methods = ['POST'])
def ClassifyObject():
    LabelCode = {0: "Картон", 1: "Скло", 2: "Метал", 3: "Папір", 4: "Пластик", 5: "Сміття"}
    LabelCode = {value: key for key, value in LabelCode.items()}
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    image_base64 = base64.b64encode(file.read()).decode('utf-8')
    result, accuracy, recomms = ObjClassification(file.stream)
    category_id = LabelCode[result] + 1
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO Images (image, class, confidence, category_id) VALUES (?, ?, ?, ?)",
        (image_base64, result, accuracy, category_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"result": result, "accuracy": round(float(accuracy), 2), "recomms": recomms})

@app.route('/fetchall', methods = ['POST'])
def FetchAll():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT class, confidence FROM Images")
    rows = cursor.fetchall()
    conn.close()
    return jsonify({"result": [list(row) for row in rows]})
