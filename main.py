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

	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

		img_byte_arr = io.BytesIO()
		isolated.save(img_byte_arr, format='JPEG')
	
		return isolated.getvalue()
	else:
		return None

@app.post("/preprocess_image/")
async def preprocess_image(file: UploadFile = File(...)):
	contents = await file.read()
	decoded_img = img_decode(contents)
	preprocessed_img = preprocessimage(decoded_img)
	encoded_img = img_encode(preprocessed_img)
	if preprocessed_img:
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
	img = Image.open(io.BytesIO(image))
	width, height = img.size

	img1 = img.crop((0, 0, width // 2, height))
	img2 = img.crop((width // 2, 0, width, height))

	img1_byte_arr = io.BytesIO()
	img2_byte_arr = io.BytesIO()

	img1.save(img1_byte_arr, format='JPEG')
	img2.save(img2_byte_arr, format='JPEG')

	return img1_byte_arr.getvalue(), img2_byte_arr.getvalue()

def concat_images(image1, image2):
	img1 = Image.open(io.BytesIO(image1))
	img2 = Image.open(io.BytesIO(image2))

	total_width = img1.width + img2.width
	max_height = max(img1.height, img2.height)

	new_img = Image.new('RGB', (total_width, max_height))
	new_img.paste(img1, (0, 0))
	new_img.paste(img2, (img1.width, 0))

	img_byte_arr = io.BytesIO()
	new_img.save(img_byte_arr, format='JPEG')
	
	return img_byte_arr.getvalue()


@app.post("/preprocess_concat_image/")
async def preprocess_image(file: UploadFile = File(...)):
	contents = await file.read()
	try:
		decoded_img = img_decode(contents)
		img1_data, img2_data = split_images(decoded_img)
		preprocessed_img1 = preprocessimage(img1_data)
		preprocessed_img2 = preprocessimage(img2_data)
		concatenated_image = concat_images(preprocessed_img1, preprocessed_img2)
		encoded_img = img_encode(concatenated_image)
		return StreamingResponse(io.BytesIO(encoded_img), media_type="image/jpeg")
	except Exception as e:
		return {"message": f"An error occurred: {e}"}