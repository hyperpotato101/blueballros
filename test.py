from keras.models import model_from_json
import cv2
import numpy as np
import os 
import random
from keras.models import Sequential


json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("blue_ball.h5")

loaded_model.compile(loss='binary_crossentropy',optimizer = 'Adam',metrics = ['accuracy'])

cap = cv2.VideoCapture(0)
while True:
	ret,frame = cap.read()
	if ret:
		font = cv2.FONT_HERSHEY_SIMPLEX
		n = cv2.resize(frame,(400,400))
		X = np.array(n).reshape(-1,400,400,3)
		X = X/255.0
		y = loaded_model.predict(X)
		print(y)
		if(y>=0.5):
			cv2.putText(frame,' Blueball', (0,150),font,1,(255,0,0),2)
			cv2.imshow("output",frame)
		else:
			cv2.putText(frame,' NOT Blueball', (0,150),font,1,(0,0,255),2)
			cv2.imshow("output",frame)
	else:
		break

	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllwindows()