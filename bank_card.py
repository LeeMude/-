import cv2 as cv
import numpy as np
from imutils import contours#排序操作

'''
第一步，处理数字模板
'''
#绘图展示
def show(name,img):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#数字模板图片
img_num = cv.imread('C:\images\ocr_a_reference.png')
#数字模板灰度图
img_num_gray = cv.cvtColor(img_num,cv.COLOR_BGR2GRAY)
#二值化  该函数返回两值，第一个值是设定的阈值，第二个返回的是二值化后的图像 二值化图像背景为黑色，识别的对象为白色
#threshold,img_num_binary = cv.threshold(img_num_gray,10,255,cv.THRESH_BINARY_INV)
#二值化还可以这么写，后面的[1]表示取返回值的第二个，即二值化的图像
img_num_binary = cv.threshold(img_num_gray,10,255,cv.THRESH_BINARY_INV)[1]

#show("img_num",img_num)
#show("img_num",img_num_gray)
show("img_num",img_num_binary)

#对数字模板进行轮廓检测 返回轮廓位置的数组
binary_contours = cv.findContours(img_num_binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[1]
print(len(binary_contours))

#排列轮廓的顺序
binary_contours = sorted(binary_contours, key=lambda x: x[0][0][0], reverse=False)
# print(binary_contours[0])
# print(binary_contours[0][0])
# print(binary_contours[0][0][0])
#show("1",binary_contours)
#绘制在原图上绘制轮廓
draw_img = img_num.copy()
cv.drawContours(draw_img,binary_contours,-1,(0,0,255),2)
show("draw_img",draw_img)

#把数字抠出来
digits = {}#空字典存储外接矩形
for (i,c) in enumerate(binary_contours):
    (x,y,w,h) = cv.boundingRect(c)
    roi = img_num_binary[y:y+h,x:x+w]
    cv.resize(roi,(57,88))
    digits[i]=roi

#show('1',digits[5])
'''
第二步，处理银行卡
'''
#读入银行卡
card = cv.imread('C:\images\credit_card_01.png')
show("card",card)
#转化为灰度图
card_gray = cv.cvtColor(card,cv.COLOR_BGR2GRAY)
show("card_gray",card_gray)
card_binary = cv.threshold(card_gray,127,255,cv.THRESH_BINARY)[1]#返回值中第一个是设定的阈值，第二个才是二值化图像所以后面加[1]
#show('binary',card_binary)

#卷积核
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
rectKernel = cv.getStructuringElement(cv.MORPH_RECT,(9,3))

#顶帽操作，突出更明亮的区域
tophat = cv.morphologyEx(card_gray,cv.MORPH_TOPHAT,kernel)
show('tophat',tophat)

#梯度运算，突出边界
gradx = cv.Sobel(tophat,ddepth=cv.CV_32F,dx=1,dy=0,ksize=-1)
#在梯度运算中，黑到白为正值，白到黑为负值，所以要加绝对值
gradx = np.absolute(gradx)
#把运算的值转化为0到255
(minVal,maxVal)=(np.min(gradx),np.max(gradx))
gradx = (255*((gradx-maxVal)/(maxVal-minVal)))
#进行梯度运算的时候是32位浮点型，现在要转换成8位无符号整点型
gradx = gradx.astype('uint8')
show('gradx',gradx)

#目的是得出一串数字的位置和大小，所以运用闭运算使几个数字的区域全部变白色
gradx = cv.morphologyEx(gradx,cv.MORPH_CLOSE,rectKernel)
show('close1',gradx)
#二值化
gradx = cv.threshold(gradx,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
show('binary',gradx)
#继续闭操作
gradx = cv.dilate(gradx,kernel,iterations=4)
gradx = cv.erode(gradx,kernel,iterations=4)
# gradx = cv.morphologyEx(gradx,cv.MORPH_CLOSE,kernel)
show('close2',gradx)

#轮廓检测
gradx_contours = cv.findContours(gradx.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) [1]
card_contours = cv.drawContours(card.copy(),gradx_contours,-1,(0,0,255),3)
show('card_contours',card_contours)

#画银行卡数字的外接矩形
locs = []#存储矩形的位置和大小
card_num = card.copy()
for (i, c) in enumerate(gradx_contours):
    (x,y,w,h) = cv.boundingRect(c)
    #print((x,y,w,h))
    # cv.rectangle(card_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # show('caed_contours', card_contours)
    k = float(w/h)
    #print(k)
    if k>3 and k<4.0:
        if (w>90 and w<100) and (h>24 and h<32):
            locs.append(((x,y,w,h)))
            cv.rectangle(card_num,(x,y),(x+w,y+h),(255,255,0),5)
#show('card_num',card_num)
#排序
#print(locs)
locs = sorted(locs,key=lambda x:x[0],reverse=False)
#print(locs)
#数字匹配
output = []
for (i,(gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []
    #提取每一组的坐标
    group = card_binary[gy-5:gy+gh,gx-5:gx+gw]
    #找数字串的轮廓
    digitcnts = cv.findContours(group,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[1]
    # print(digitcnts[0])
    # print(digitcnts[0][0])
    #排序
    digitcnts = sorted(digitcnts,key=lambda x:x[0][0][0])
    #print(digitcnts)

    #找到每一个数字
    for (i,c) in enumerate(digitcnts):
        (x,y,w,h) = cv.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv.resize(roi,(57,88))#为了与模板更好的配对
        #show('roi',roi)

        #计算匹配得分
        scores = []

        for (num,digitRoi) in digits.items():
            result = cv.matchTemplate(roi,digitRoi,cv.TM_CCOEFF)
            (min_score,max_score,min_loc,max_loc) = cv.minMaxLoc(result)
            scores.append(max_score)#把模板的全部数字与该数字roi匹配，并记录每个数字模板与该数字匹配的最高分
        #取出scores这个数组最高分的就是roi的数字了
        groupOutput.append(str(np.argmax(scores)))
    #把数字画出来
    cv.putText(card_num,' '.join(groupOutput),(gx,gy-15),cv.FONT_HERSHEY_COMPLEX,0.75,(0,0,255),2)
    output.extend(groupOutput)
print(f"card:{''.join(output)}")
show('image',card_num)




