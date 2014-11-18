import cv2
import random
import numpy as np

cv2.namedWindow("preview", cv2.WINDOW_OPENGL )
vc = cv2.VideoCapture(0)
vc.set(3,1920)
vc.set(4,1080)
face_cascade = cv2.CascadeClassifier('C:\\Users\\dherman\\Documents\\Research Projects\\Python Scripts\\Image Analysis Class\\haarcascade_frontalface_alt.xml')#haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



def Split_Blur_Thresh(frame):
	
	b,g,r = cv2.split(frame)
	b = cv2.medianBlur(b,7)
	g = cv2.medianBlur(g,7)
	r = cv2.medianBlur(r,7)
	b = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	g = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	r = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	
	return cv2.merge((b,g,r))
	
def OpenCV_Test(frame,Y_location):
	cv2.putText(frame, 'This is an OpenCV Test!!', (Y_location,frame.shape[1]/2), cv2.FONT_HERSHEY_SIMPLEX, 3, [0,0,255], thickness=4)

	return frame


def Corners(frame):
	#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html#harris-corners

	b,g,r = cv2.split(frame)
	for gray in [b,g,r]:
		gray = np.float32(b)
		dst = cv2.cornerHarris(gray,2,3,0.04)
		
		#result is dilated for marking the corners, not important
		dst = cv2.dilate(dst,None)

		# Threshold for an optimal value, it may vary depending on the image.
		frame[dst>0.01*dst.max()]=[0,0,255]
	return frame
	
	
	
def CannyEdge(frame,color='all'):
	#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
	frame = cv2.medianBlur(frame,5)
	if color == 'all':
		b,g,r = cv2.split(frame)
		b= cv2.Canny(b,70,200)
		g= cv2.Canny(g,70,200)
		r= cv2.Canny(r,70,200)
		return cv2.merge((b,g,r))
	if color == 'b':
		b,g,r = cv2.split(frame)
		b= cv2.Canny(b,70,200)
		#g= cv2.Canny(g,70,200)
		#r= cv2.Canny(r,70,200)
		return cv2.merge((b,g,r))	
	if color == 'g':
		b,g,r = cv2.split(frame)
		#b= cv2.Canny(b,70,200)
		g= cv2.Canny(g,70,200)
		#r= cv2.Canny(r,70,200)
		return cv2.merge((b,g,r))
	if color == 'r':
		b,g,r = cv2.split(frame)
		#b= cv2.Canny(b,70,200)
		#g= cv2.Canny(g,70,200)
		r= cv2.Canny(r,70,200)
		return cv2.merge((b,g,r))
		


def FaceTime(frame):
	#http://docs.opencv.org/trunk/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
	#frame = cv2.medianBlur(frame,5)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Find Faces
	faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
	
	#Debugging
	#print face_cascade,faces
	#http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html
	if len(faces)== 2:
		#Get Face images
		face_1= frame[ faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2] ]
		face_2= frame[ faces[1][1]:faces[1][1]+faces[1][3],faces[1][0]:faces[1][0]+faces[1][2] ]
		#Rescale
		face_2_rescale=cv2.resize( face_2, (faces[0][3],faces[0][2]) , interpolation=cv2.cv.CV_INTER_CUBIC ) 
		face_1_rescale=cv2.resize( face_1, (faces[1][3],faces[1][2]) , interpolation=cv2.cv.CV_INTER_CUBIC ) 
		#Apply faces swtich
		frame[ faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2] ]=face_2_rescale
		frame[ faces[1][1]:faces[1][1]+faces[1][3],faces[1][0]:faces[1][0]+faces[1][2] ]=face_1_rescale
	elif len(faces)== 1:
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)	
	else:
		pass
		#Add additional functionality
	return frame
	
	
if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	'Camera is not working?'
	rval = False

	
Y_location=20	
while rval:
	#Show new frame in window
	cv2.imshow("preview", frame)
	#capture next image from webcam
	rval, frame = vc.read()
	
	
	#OpenCV test function
	'''
     frame=OpenCV_Test(frame,Y_location)
	Y_location+=5
	if Y_location > frame.shape[0]-20:
		Y_location=20
	'''	
	#Advanced Functions
	
	#frame=CannyEdge(frame,color='all')
	frame = Split_Blur_Thresh(frame)
	#frame=Corners(frame)
	#frame=FaceTime(frame)
	
	#Check for userinput
	key = cv2.waitKey(10)
	if key == 27: # exit on ESC
		vc.release
		break
	elif key =='s' :#Save image
		cv2.imwrite('C:\\Users\\Public\\Pictures\\OpenCV_Testimage.png', frame)
print 'close early? check for earlier run runmning'		
		
		
cv2.destroyWindow("preview")
vc.release