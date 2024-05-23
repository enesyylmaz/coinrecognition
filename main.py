from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import json

app = FastAPI()

with open('coin_information.json') as f:
	data = json.load(f)

def preprocessimage(image):
	offset = 1
	param2Value = 110
	param2Change = 7

	org = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
	image = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
	image = cv2.blur(image, (7,7))
	orgGray = np.array(image, copy=True)

	height, width = image.shape[:2]
	circles = None
	while (circles is None) and (param2Value > 50) :
		image = np.array(orgGray, copy=True)
		circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,40,
								param1=60,param2=param2Value,minRadius=0,maxRadius=0)
		param2Value -= param2Change
	
	if circles is not None:
		original = np.array(circles, copy=True)
		circles = np.uint16(np.around(circles))
		mask = np.zeros_like(image)

		maxCircle = max(circles[0, :], key = lambda x: x[2])
		radius = int(maxCircle[2] * offset)
		cv2.circle(mask, (maxCircle[0], maxCircle[1]), radius, (255, 255, 255), thickness=-1)
	
		top = int(maxCircle[1] - radius)
		bottom = int(maxCircle[1] + radius)
		left = int(maxCircle[0] - radius)
		right = int(maxCircle[0] + radius)
		
		result = cv2.bitwise_and(org, org, mask=mask)
		isolated = result[top:bottom, left:right]

		isolated = cv2.resize(isolated, (256,256))
		  
		_, img_encoded = cv2.imencode('.jpg', isolated)
		return img_encoded.tobytes()
	else:
		return None

@app.post("/preprocess_image/")
async def preprocess_image(file: UploadFile = File(...)):
	contents = await file.read()
	preprocessed_img = preprocessimage(contents)
	if preprocessed_img:
		return StreamingResponse(io.BytesIO(preprocessed_img), media_type="image/jpeg")
	else:
		return {"message": "No circles detected in the image."}

@app.get("/get_info/{key}")
async def get_info(key: str):
	for entry in data["entries"]:
		if key in entry:
			return entry[key]
	return {"message": "Key not found in the data."}


def split_images(image_array):
    image = Image.open(io.BytesIO(image_array))
    width, height = image.size
    img1 = image.crop((0, 0, width // 2, height))
    img2 = image.crop((width // 2, 0, width, height))
    img1_byte_array = io.BytesIO()
    img1.save(img1_byte_array, format="JPEG")
    img2_byte_array = io.BytesIO()
    img2.save(img2_byte_array, format="JPEG")
    return img1_byte_array.getvalue(), img2_byte_array.getvalue()

def concat_images(image1, image2):
    img1 = Image.open(io.BytesIO(image1))
    img2 = Image.open(io.BytesIO(image2))
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    concatenated_image = Image.new('RGB', (total_width, max_height))
    concatenated_image.paste(img1, (0, 0))
    concatenated_image.paste(img2, (img1.width, 0))
    byte_array = io.BytesIO()
    concatenated_image.save(byte_array, format="JPEG")
    return byte_array.getvalue()

@app.post("/preprocess_concat_image/")
async def preprocess_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img1_data, img2_data = split_images(contents)
        preprocessed_img1 = preprocessimage(img1_data)
        preprocessed_img2 = preprocessimage(img2_data)
        concatenated_image = concat_images(preprocessed_img1, preprocessed_img2)
        return StreamingResponse(io.BytesIO(concatenated_image), media_type="image/jpeg")
    except Exception as e:
        return {"message": f"An error occurred: {e}"}