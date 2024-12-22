import dlib
import cv2
predictor = "shape_predictor_5_face_landmarks.dat"  #模型(5點)
sp = dlib.shape_predictor(predictor)  #讀入模型
detector = dlib.get_frontal_face_detector()  #偵測臉部

img = dlib.load_rgb_image("media\\LINE_230625.jpg")  #讀取圖片
win = dlib.image_window()  #建立顯示視窗
win.clear_overlay()  #清除圖形
win.set_image(img)  #顯示圖片

dets = detector(img, 1)  #臉部偵測,1為彩色
print("人臉數：{}".format(len(dets)))
#繪製人臉矩形及5點特徵
for k, det in enumerate(dets):
    print("偵測人臉 {}: 左：{}  上：{}  右：{}  下：{}".format(k, det.left(), det.top(), det.right(), det.bottom()))  #人臉坐標
    win.add_overlay(det)  #顯示矩形
    shape = sp(img, det)  #取得5點特徵 
    win.add_overlay(shape)  #顯示5點特徵
    dlib.hit_enter_to_continue()  #保持影像