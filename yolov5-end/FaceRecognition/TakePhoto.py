import cv2
import numpy as np
import time

# 鱼眼有效区域截取
def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
    cv2.imshow("binary",thresh)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # con = img
    # con = cv2.drawContours(con,contours,-1,(0,0,255),3)
    # cv2.namedWindow('contours',cv2.WINDOW_NORMAL)
    # cv2.imshow("contours",con)
    # cv2.waitKey()
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # # print(cnts)
    # con = img
    # con = cv2.drawContours(con,cnts,-1,(0,0,255),3)
    # cv2.namedWindow('cnts',cv2.WINDOW_NORMAL)
    # cv2.imshow("cnts",con)
    # cv2.waitKey()
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)
    # 提取有效区域
    img_valid = img[y:y+h, x:x+w]
    print(x,y,h,w)
    return img_valid, int(r)

# 鱼眼矫正
def undistort(src,r):
    # r： 半径， R: 直径
    R = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape

    # 圆心
    x0, y0 = src_w//2, src_h//2

    # 数组， 循环每个点
    range_arr = np.array([range(R)])

    theta = Pi - (Pi/R)*(range_arr.T)
    temp_theta = np.tan(theta)**2

    phi = Pi - (Pi/R)*range_arr
    temp_phi = np.tan(phi)**2

    tempu = r/(temp_phi + 1 + temp_phi/temp_theta)**0.5
    tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

    # 用于修正正负号
    flag = np.array([-1] * r + [1] * r)

    # 加0.5是为了四舍五入求最近点
    u = x0 + tempu * flag + 0.5
    v = y0 + tempv * np.array([flag]).T + 0.5

    # 防止数组溢出
    u[u<0]=0
    u[u>(src_w-1)] = src_w-1
    v[v<0]=0
    v[v>(src_h-1)] = src_h-1

    # 插值
    dst[:, :, :] = src[v.astype(int),u.astype(int)]
    return dst

def snapShotCt(user_name): # camera_idx的作用是选择摄像头。如果为0则使用内置摄像头，比如笔记本的摄像头，用1或其他的就是切换摄像头。
    # cv2.namedWindow(user_name,cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    cap.set(3,1920)
    cap.set(4,1280)
    while True:
        ret, frame = cap.read() # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow(user_name, small_frame) # 写入图片
        if cv2.waitKey(5) & 0xFF == ord(' '):
            cv2.imwrite(user_name+".jpg", frame) # 写入图片
            break
    cap.release() # 释放

if __name__ == "__main__":
    user_name = input("请输入用户名/文件名：")
    print("按[空格键]拍摄")
    snapShotCt(user_name)
    frame = cv2.imread(user_name+'.jpg')
    cut_img,R = cut(frame)
    result_img = undistort(cut_img,R)
    cv2.imwrite('user_'+user_name+'.jpg',result_img)
