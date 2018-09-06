import dlib
import cv2
import glob
import tqdm
import pandas as pd
import scipy
import numpy as np
import scipy.io
###########################################################################
# paths = glob.glob(r'images\single_people\**\*.jpg')
paths = glob.glob(r'..\utk\UTKface\**\*.jpg')
# print(paths)
detector = dlib.get_frontal_face_detector()
###################################################################################
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

################ For learning #################################
# arg = argparse.ArgumentParser()
# arg.add_argument('-n' , '--name', required = True , type = str, help='name of the user')
# args = vars(arg.parse_args())
# print('Hi {}'.format(args['name']))
###################################################################################
i = 1
ages = []
genders = []
races = []
faces = []
margin = 10
for path in paths:
	img = cv2.imread(path)
	img = cv2.resize(img , (500, 500))
	width , height , ch = img.shape
	# print(width , height)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	rects = detector(gray , 1)
	face = []
	# print(rects)
	try:
		if len(rects) > 0  :
			for (i, rect) in enumerate(rects):
				# print(path)
				if (len(path.split('\\')[4].split('_')) == 4):
					name= path.split('\\')[4].split('_')[:3]
					age  = int(name[0])
					gender = int(name[1])
					race = int(name[2])
					(x,y,w,h) = rect_to_bb(rect)
					# print(rect)
					if(x > margin and y > margin and x+w < width - margin and y+h < height - margin ):
						# cv2.rectangle(img,(x-margin,y-margin),(x+w+margin,y+h+margin),(255,0,0),2)
						face = cv2.resize( img[y-margin:y+h+margin, x-margin:x+w+margin], (64 , 64))
						faces.append(face)
						ages.append(age)
						genders.append(gender)
						races.append(race)
						print(int(name[0]), int(name[1]) , int(name[2]))
						# cv2.imshow('Img', img)
						# cv2.imshow('Face', face)
		else :
			continue
	except :
		continue
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
	if k == ord('m'):
		continue
	# break

# flattened_faces = np.array(faces).flatten().T
print('faces:',len(faces) ,'age:',  len(ages),'gender:', len(genders) , 'race:', len(races))
# output = {'image':flattened_faces , 'gender':np.array(gender) , 'age': np.array(age) , 'race': np.array(race)}
# df = pd.DataFrame(data = output)
# df.to_csv = ('images\\data.csv')
output = {'image':faces, 'gender':np.array(genders) , 'age': np.array(ages) , 'race': np.array(races)}
scipy.io.savemat('images\\data_base.mat' , output )
cv2.destroyAllWindows()


