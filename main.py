import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys
import json
import keras as K

from fastapi import Response

from starlette.responses import StreamingResponse


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Загружаем модель
filepath = 'model/model_1.h5'
model = load_model(filepath, compile=True)
filepath_s = 'model/model_segm.h5'
model_s = load_model(filepath_s, compile=True,
                     custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

app = FastAPI()


class Prediction(BaseModel):
    filename: str
    contenttype: str
    prediction: List[float] = []
    likely_class: str


@app.get('/')
def root_route():
    return {'error': 'Use GET /prediction instead of the root route!'}


@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):
    # Убедиться, что это изображение
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Читать содержимое изображения
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Преобразование  RGBA  RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Изменить размер изображения до ожидаемой входной формы
        pil_image = pil_image.resize((224, 224))

        #  Преобразование изображения
        numpy_image = np.asarray(pil_image)

        # прогноз классификации
        prediction_array = np.array([numpy_image])
        predictions = model.predict(prediction_array)
        prediction = predictions[0]

        # Opening JSON file
        label = open('labels.json', encoding="UTF-8")
        data = json.load(label)

        label.close()

        if np.amax(prediction) < 0.7:
            likely_class_pr = "не определен"
        else:
            likely_class_pr = data[str(np.argmax(prediction))]

        return {
            'filename': file.filename,
            'contenttype': file.content_type,
            'prediction': prediction.tolist(),
            'likely_class': likely_class_pr
        }
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/segmintation/', response_class=Response)
async def prediction_route(file: UploadFile = File(...)):
    # Убедиться, что это изображение
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Читать содержимое изображения
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Преобразование  RGBA  RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Изменить размер изображения до ожидаемой входной формы
        pil_image = pil_image.resize((224, 224))

        #  Преобразование изображения
        numpy_image_s = np.array(pil_image) / 255.0

        # прогноз
        predict = np.zeros((224, 224, 3))

        for i in range(3):
            predict_ = model_s.predict(np.expand_dims(numpy_image_s, axis=0))[i]
            predict_ = np.squeeze(predict_, axis=0)
            predict_ = np.squeeze(predict_, axis=2)
            predict[:, :, i] = predict_

        predict_img = predict * 255

        print(predict_img.shape)

        print(predict_img)

        data = Image.fromarray((predict_img * 1).astype(np.uint8)).convert('RGB')

        buf = io.BytesIO()
        data.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        image_stream = io.BytesIO(byte_im)
        return StreamingResponse(content=image_stream, media_type="image/jpeg")



    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
