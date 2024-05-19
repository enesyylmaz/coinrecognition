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
	image = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
	image = cv2.blur(image, (7, 7))
	orgGray = np.array(image, copy=True)

	height, width = orgGray.shape[:2]
	circles = None
	while (circles is None) and (param2Value > 50):
		image = np.array(orgGray, copy=True)
		circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 40,
									param1=60, param2=param2Value, minRadius=0, maxRadius=0)
		param2Value -= param2Change

	if circles is not None:
		original = np.array(circles, copy=True)
		circles = np.uint16(np.around(circles))
		mask = np.zeros_like(image)

		maxCircle = max(circles[0, :], key=lambda x: x[2])
		radius = int(maxCircle[2] * offset)
		cv2.circle(mask, (maxCircle[0], maxCircle[1]), radius, (255, 255, 255), thickness=-1)

		top = int(maxCircle[1] - radius)
		bottom = int(maxCircle[1] + radius)
		left = int(maxCircle[0] - radius)
		right = int(maxCircle[0] + radius)

		result = cv2.bitwise_and(org, org, mask=mask)
		isolated = result[top:bottom, left:right]
		
		isolated = cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB)
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