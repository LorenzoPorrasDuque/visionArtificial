from pyexpat import features
import cv2
import numpy as np
from glob import glob
import xlsxwriter

vector_folders_nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
row=0
workbook=xlsxwriter.Workbook('/home/anime/Desktop/visionArtificial/extract/caractNums.xlsx')
worksheet=workbook.add_worksheet('caracts')
vector_caracts=np.array([])

def binarizeImg(imgGray):
    return cv2.inRange(imgGray, 0, 127)
def extractContours(imgBinary):
    contours, _ = cv2.findContours(imgBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt=[]
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
    return cnt

def extractPatterns(imgRoiBin,cnt):
    x1_area=cv2.contourArea(cnt)
    x2_perimeter=cv2.arcLength(cnt,True)
    x3_circularity=x1_area/(x2_perimeter*x2_perimeter)
    M=cv2.moments(cnt)
    Hu=cv2.HuMoments(M)
    x4_Hu1=Hu[0][0]
    x5_Hu2=Hu[1][0]
    x6_Hu3=Hu[2][0]
    x7_Hu4=Hu[3][0]
    x8_Hu5=Hu[4][0]
    x9_Hu6=Hu[5][0]
    x10_Hu7=Hu[6][0]
    roi1= imgRoiBin[0:10,0:10]/100
    count_bin = cv2.countNonZero(roi1)/100
    x11=cv2.countNonZero(roi1)/100

    list_car=[x1_area,x2_perimeter,x3_circularity,x4_Hu1,x5_Hu2,x6_Hu3,x7_Hu4,x8_Hu5,x9_Hu6,x10_Hu7,x11]
    features=np.array(list_car)
    return features
def extractCaracts():
    
    global row,vector_caracts
    for n in range(0,len (vector_folders_nums)):
        for imgPath in glob("/home/anime/Desktop/visionArtificial/extract/num/"+vector_folders_nums[n]+"/*.png"):
            imgGray = cv2.imread(imgPath,0)
            imgColor = cv2.imread(imgPath,1)

            imgBinary=binarizeImg(imgGray)


            cnt= extractContours(imgBinary)
            if len(cnt) > 0:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(imgColor,(x,y),(x+w,y+h),(255,0,0),1)
                cv2.imshow("imgColor",cv2.resize(imgColor,(200,200)))
                imgRoiBin=imgBinary[y:y+h,x:x+w]
                imgRoiBin=cv2.copyMakeBorder(imgRoiBin,2,2,2,2,cv2.BORDER_CONSTANT,value=0)
                imgRoiBin=cv2.resize(imgRoiBin,(20,60))
                vector_caracts=extractPatterns(imgRoiBin,cnt)
                # Write the label (digit) in first column
                worksheet.write(row,0,int(vector_folders_nums[n]))
                # Write all features in subsequent columns
                for i, caract in enumerate(vector_caracts):
                    worksheet.write(row,i+1,float(caract))
                row+=1
            else:
                cv2.waitKey(1)


            cv2.imshow("imgBinary",cv2.resize(imgBinary,(200,200)))
            cv2.waitKey(1)
        cv2.destroyAllWindows()
    


extractCaracts()
workbook.close()