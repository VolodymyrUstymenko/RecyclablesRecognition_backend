import numpy as np
import sqlite3
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from skimage import io as skio, color

def GetRecommendations(class_name):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT recomendations FROM Categories WHERE name = ?", (class_name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Категорію не знайдено"

def ObjClassification(image_stream):
    LabelCode = {0: "Картон", 1: "Скло", 2: "Метал", 3: "Папір", 4: "Пластик", 5: "Сміття"}
    image = skio.imread(image_stream)
    image = resize(image, (256, 256)) / 255.0
    model = load_model('CNN_model.keras')
    image = np.expand_dims(image, axis = 0)
    probs = model.predict(image)
    preds = probs.argmax(axis=1)[0]
    result = LabelCode[preds]
    accuracy = float(probs[0][preds])
    recomms = GetRecommendations(result)
    return result, accuracy, recomms