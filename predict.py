import numpy as np
import dlib
import cv2
import my_model

race_dict={'White':0 , 'Black':1 , 'Asian':2 , 'Indian':3 , 'Others':4}
race_inv_dict = {i:v for v , i in zip(race_dict.keys() , race_dict.values())}

gender_dict={'Male': 0 , 'Female': 1}
gender_inv_dict = {i:v for v , i in zip(gender_dict.keys() , gender_dict.values())}

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

detector = dlib.get_frontal_face_detector()

img = cv2.imread('test\\8.jpg')
model = my_model.My_Model(weights_path = 'recognition_age_3.h5')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rects = detector(gray , 1)
margin = 10
faces = []
if len(rects)>0:
	# for  i in range(len(rects)):
	(x,y,w,h) = rect_to_bb(rects[0])
	cv2.rectangle(img,(x-margin,y-margin),(x+w+margin,y+h+margin),(255,0,0),2)
	face = cv2.resize( img[y-margin:y+h+margin, x-margin:x+w+margin], (64 , 64))
	faces.append(face)
		
faces = np.array(faces)
print(faces.shape)
g , a ,r = model.predict(faces)
print(g.shape , a.shape , r.shape)
a = a.T
g = g.T
r = r.T
print(g.shape , a.shape , r.shape)
age = np.argmax(a)
gender = np.argmax(g)
race = np.argmax(r)
print(age+1 ,gender , race)
print(age + 1 , gender_inv_dict[gender]  , race_inv_dict[race])

cv2.putText(img, 'Age:'+str(age+1) ,org= (100,100), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 1 ,color =(255,0,0), thickness = 2  )
cv2.putText(img, str(gender_inv_dict[gender]) ,org= (100,200), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 1 ,color =(255,0,0), thickness = 2 )
cv2.putText(img, str(race_inv_dict[race]) ,org= (100,300), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 1 ,color =(255,0,0), thickness = 2 )
				
cv2.imwrite('result\\res13.jpg' , img)				
cv2.imshow('Img' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()