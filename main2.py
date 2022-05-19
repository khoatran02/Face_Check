from datetime import datetime
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np


# step 1 load images
path ="D:\Document\Code\Python_OpenCV\Face_Check\pic2"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}") # pic2/cl
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # splitext sẽ tách path ra 2 phần, phần trước đuôi mở rộng và phần mở rộng

# step encoding
def Mahoa(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR -> RGB
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnow = Mahoa(images)
print("Ma hoa thanh cong")
print(len(encodelistKnow))

def thamdu(name):
    with open("Check.csv","r+") as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(";")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%d/%m/%Y, %H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")

# Open WebCam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frameS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    framS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

    # xác định vi trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS) # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceloc in zip(encodecurFrame,facecurFrame): # lấy từng khuôn mặt và vị trí khuôn mặt theo cặp
        matches = face_recognition.compare_faces(encodelistKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) # return index của facedis min

        if faceDis[matchIndex] < 0.45:
            name = classNames[matchIndex].upper()
            thamdu(name)
        else:
            name = "UnKnow"

        # print name lên frame
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

  #      frame = cv2.cvtColor(frame,cv2.COLORMAP_OCEAN)
    cv2.imshow("Check face",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()