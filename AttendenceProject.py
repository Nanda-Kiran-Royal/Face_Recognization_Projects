import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'ImagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)

    classNames.append(os.path.splitext(cls)[0])

print(classNames )


def findEncodings(images):
    encodeList = []
    for img in images :
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')


        print(myDataList)





encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv.resize(img,(0,0),None,.25,.25)
    imgS = cv.cvtColor(imgS,cv.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)

        print(faceDist)
        matchIndex =  np.argmin(faceDist)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+12,y2-6),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2 )
            markAttendence(name)



    cv.imshow('Webcam',img)
    cv.waitKey(1)





#
# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDist = face_recognition.face_distance([encodeElon],encodeTest)

imgElon = face_recognition.load_image_file('ImagesAttendence/ElonMusk.jpg')
imgElon = cv.cvtColor(imgElon,cv.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('gettyimages-1229892983-square.jpg')
imgTest = cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)