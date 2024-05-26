from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import json
from PIL import Image

app = FastAPI()

with open('coin_information.json') as f:
    data = json.load(f)

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

@app.post("/preprocess_image/")
async def preprocess_image(file: UploadFile = File(...)):
    contents = await file.read()
    decoded_img = img_decode(contents)
    preprocessed_img = preprocessimage(decoded_img)
    if preprocessed_img is not None:
        encoded_img = img_encode(preprocessed_img)
        return StreamingResponse(io.BytesIO(encoded_img), media_type="image/jpeg")
    else:
        return {"message": "No circles detected in the image."}

@app.get("/get_info/{key}")
async def get_info(key: str):
    for entry in data["entries"]:
        if key in entry:
            return entry[key]
    return {"message": "Key not found in the data."}

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
        return {"message": f"An error occurred: {e}"}