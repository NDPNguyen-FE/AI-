

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate


SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True

def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:                               # nếu train không thành công
        print("\nerror: KNN traning was not successful\n")  # thông báo lỗi
        return                                                          # kết thúc chương trình


    imgOriginalScene  = cv2.imread("1.jpg")               # mở image được chọn

    if imgOriginalScene is None:                            # nếu hình ảnh không được nhận diện thành công
        print("\nerror: image not read from file \n\n")  # in ra màn hình
        os.system("pause")                                  # tạm dừng chương trình
        return                                              # kết thúc chương trình
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # phát hiện image

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # tìm thấy kí tự trong image

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # hiển thị image

    if len(listOfPossiblePlates) == 0:                          # nếu không tìm thấy ký tự
        print("\nno license plates were detected\n")  # thông báo trên màn hình không tìm thấy hình
    else:



        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)


        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           # hiển thị tấm biển số
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # nếu không tìm thấy ký tự trên biển số
            print("\nno characters were detected\n\n")  # hiện thị tin nhắn
            return                                          # kết thúc chương trình


        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # khoanh viền đỏ xác định định biển số xe

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # in ra biển số dưới console
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # hiện thị biển số dưới dạng text trên ảnh

        cv2.imshow("imgOriginalScene", imgOriginalScene)                # show lại hình hảnh toàn cảnh

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # ghi hình ảnh ra



    cv2.waitKey(0)

    return



def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # vẽ 4 đường màu đỏ
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)



def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)


    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))


    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))


    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)



if __name__ == "__main__":
    main()


















