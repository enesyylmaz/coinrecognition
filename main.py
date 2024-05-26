import os
import json
import asyncio
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from aiohttp import ClientSession
import aiofiles
import cv2

app = FastAPI()

current_directory = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_directory, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

with open('coin_information.json') as f:
    data = json.load(f)

class_names = ['10150', '10151', '10250', '10251', '10270', '10271', '10280', '10281', '10290', '10291', '10410', 
               '10411', '10560', '10561', '10780', '10781', '11040', '11041', '11050', '11060', '11061', '11370', 
               '11640', '11641', '11650', '11651', '11660', '11661', '11670', '11671', '11680', '11681', '11690', 
               '11691', '12390', '12391', '12400', '12410', '12411', '12420', '12430', '12431', '12440', '5960', 
               '6670', '6700', '6751', '7350', '7351', '7360', '7370', '7371', '7440', '7441', '7480', '7481', '7490', 
               '7500', '7501', '7590', '7600', '7601', '7610', '7611', '7790', '7791', '8300', '8310', '8320', '8321', 
               '8330', '8331', '8600', '8601', '8610', '8620', '8621', '8630', '8640', '8650', '8651', '8661', '8670', 
               '8671', '8680', '8681', '8690', '8691', '8800', '8801', '8810', '8811', '8820', '8830', '8831', '8840', 
               '8841', '8850', '8851', '8860', '8861', '8870', '8880', '8890', '8891', '8900', '8901', '8910', '8911', 
               '8920', '8921', '8930', '8931', '8940', '8941', '8950', '8990', '8991', '9050', '9051', '9430', '9431', 
               '9440', '9441', '9450', '9451', '9630', '9640', '9641', '9650', '9651', '9660', '9670', '9750', '9870', 
               '9871', '9880', '9890', '9891', '9970']

def img_decode(image):
    return cv2.imdecode(np.frombuffer(image, np.uint8), -1)

def img_encode(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    return img_encoded.tobytes()

def preprocessimage(image):
    offset = 1
    param2Value = 110
    param2Change = 7

    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image1 = cv2.blur(image1, (7, 7))
    orgGray = np.array(image1, copy=True)

    height, width = image.shape[:2]
    circles = None
    while (circles is None) and (param2Value > 50):
        image1 = np.array(orgGray, copy=True)
        circles = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 40,
                                   param1=60, param2=param2Value, minRadius=0, maxRadius=0)
        param2Value -= param2Change

    if circles is not None:
        original = np.array(circles, copy=True)
        circles = np.uint16(np.around(circles))
        mask = np.zeros_like(image1)

        maxCircle = max(circles[0, :], key=lambda x: x[2])
        radius = int(maxCircle[2] * offset)
        cv2.circle(mask, (maxCircle[0], maxCircle[1]), radius, (255, 255, 255), thickness=-1)

        top = int(maxCircle[1] - radius)
        bottom = int(maxCircle[1] + radius)
        left = int(maxCircle[0] - radius)
        right = int(maxCircle[0] + radius)

        result = cv2.bitwise_and(image, image, mask=mask)
        isolated = result[top:bottom, left:right]

        isolated = cv2.resize(isolated, (256, 256))

        return isolated
    else:
        return None

def split_images(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = pil_image.size

    img1 = pil_image.crop((0, 0, width // 2, height))
    img2 = pil_image.crop((width // 2, 0, width, height))

    return img1, img2

def concat_images(image1, image2):
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)

    new_img = Image.new('RGB', (total_width, max_height))
    new_img.paste(image1, (0, 0))
    new_img.paste(image2, (image1.width, 0))

    return new_img

async def send_to_model(image_np):
    data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
    headers = {"content-type": "application/json"}
    async with ClientSession() as session:
        async with session.post('https://coin-model-7ynk.onrender.com/v1/models/coin_model:predict', data=data, headers=headers) as response:
            return await response.json()

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filename = 'image.jpg'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        image = Image.open(filepath)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file")

    img_height, img_width = 256, 256

    rotated_image = image.rotate(-90, expand=True)
    rotated_filepath = os.path.join(UPLOAD_FOLDER, 'rotated_' + filename)
    rotated_image.save(rotated_filepath)

    min_dimension = min(rotated_image.width, rotated_image.height)
    left = (rotated_image.width - min_dimension) / 2
    top = (rotated_image.height - min_dimension) / 2
    right = left + min_dimension
    bottom = top + min_dimension
    cropped_rotated_image = rotated_image.crop((left, top, right, bottom))

    cropped_rotated_filepath = os.path.join(UPLOAD_FOLDER, 'cropped_rotated_' + filename)
    cropped_rotated_image.save(cropped_rotated_filepath)

    resized_image = cropped_rotated_image.resize((img_width, img_width))

    reduced_quality_filepath = os.path.join(UPLOAD_FOLDER, f"reduced_quality_{filename}")
    reduced_quality_image = resized_image.copy()
    reduced_quality_image.save(reduced_quality_filepath, quality=90)

    # Read the reduced quality file content
    async with aiofiles.open(reduced_quality_filepath, 'rb') as f:
        file_content = await f.read()

    # Decode the image
    decoded_img = img_decode(file_content)

    # Preprocess the image
    preprocessed_img = preprocessimage(decoded_img)
    if preprocessed_img is None:
        raise HTTPException(status_code=400, detail="No circles detected in the image.")

    # Convert the preprocessed image to a format suitable for model prediction
    preprocessed_img_pil = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
    image_np = np.array(preprocessed_img_pil)
    image_np = image_np.reshape((1, img_width, img_width, 3))

    np.save(os.path.join(UPLOAD_FOLDER, "image_np.npy"), image_np)

    predictions = await send_to_model(image_np)
    predicted_class = class_names[np.argmax(predictions['predictions'][0])]

    key = predicted_class
    coin_info = await get_info(key)

    return JSONResponse(content={
        "message": "File uploaded successfully",
        "filename": 'cropped_rotated_' + filename,
        "reduced_quality_filename": f"reduced_quality_{filename}",
        "predicted_class": predicted_class,
        "coin_info": coin_info
    })

@app.post("/preprocess_image/")
async def preprocess_image(file: UploadFile = File(...)):
    contents = await file.read()
    decoded_img = img_decode(contents)
    preprocessed_img = preprocessimage(decoded_img)
    if preprocessed_img is not None:
        encoded_img = img_encode(preprocessed_img)
        return StreamingResponse(BytesIO(encoded_img), media_type="image/jpeg")
    else:
        return JSONResponse(content={"message": "No circles detected in the image."}, status_code=400)

@app.get("/get_info/{key}")
async def get_info(key: str):
    for entry in data["entries"]:
        if key in entry:
            return entry[key]
    return JSONResponse(content={"message": "Key not found in the data."}, status_code=404)


@app.post("/preprocess_concat_image/")
async def preprocess_concat_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        decoded_img = img_decode(contents)
        img1_data, img2_data = split_images(decoded_img)
        preprocessed_img1 = preprocessimage(np.array(img1_data))
        preprocessed_img2 = preprocessimage(np.array(img2_data))
        concatenated_image = concat_images(Image.fromarray(cv2.cvtColor(preprocessed_img1, cv2.COLOR_BGR2RGB)),
                                           Image.fromarray(cv2.cvtColor(preprocessed_img2, cv2.COLOR_BGR2RGB)))
        encoded_img = img_encode(np.array(concatenated_image))
        return StreamingResponse(io.BytesIO(encoded_img), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"message": f"An error occurred: {e}"}, status_code=500)