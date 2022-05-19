import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np

imgElon = face_recognition.load_image_file("pic/elon check.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgCheck = face_recognition.load_image_file("pic/elon musk.jpg")
imgCheck =cv2.cvtColor(imgCheck,cv2.COLOR_BGR2RGB)

# Xác định vị trí khuôn mặt
faceLoc = face_recognition.face_locations(imgElon)[0]
print(faceLoc)  # (y1,x2,y2,x1)
# mã hóa hình ảnh
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)


faceCheck = face_recognition.face_locations(imgCheck)[0]
encodeCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeCheck)
print(results)

# khoảng cách (sai số ) giữa các bức ảnh là bao nhiêu?
faceDis = face_recognition.face_distance([encodeElon],encodeCheck)
print(results,faceDis)

cv2.putText(imgCheck,f"{results}{(round(faceDis[0],2))}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow("Elon",imgElon)  # view thử ảnh để kiểm tra
cv2.imshow("ElonCheck",imgCheck) # view thử ảnh
cv2.waitKey()
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ
