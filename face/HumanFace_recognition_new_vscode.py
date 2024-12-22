#pip install cmake
# 安裝 opencv 以及 dlib
#pip install opencv-python dlib
import requests
import os
import tarfile

# 設定下載網址和存放位置
url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
download_path = "./lfw-deepfunneled.tgz"
extract_path = "."

# 確保目錄存在
os.makedirs(extract_path, exist_ok=True)
'''
# 下載文件
print("正在下載文件...")
response = requests.get(url, stream=True)
with open(download_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)
print("文件下載完成！")

# 解壓文件
print("正在解壓文件...")
with tarfile.open(download_path, "r:gz") as tar:
    tar.extractall(path=extract_path)
print(f"解壓完成！文件已存放在: {extract_path}")

# 刪除壓縮包（可選）
os.remove(download_path)
print("壓縮包已刪除！")
'''
import os
import cv2
import dlib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

def detect_and_mark_faces(input_dir, output_dir):
    # 檢查輸出目錄是否存在，否則建立
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化人臉檢測器
    detector = dlib.get_frontal_face_detector()

    # 遍歷圖庫目錄中的所有檔案
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)

        # 確保檔案為圖片
        if not (file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
            print(f"跳過非圖片檔案: {file_name}")
            continue

        # 讀取圖片
        image = cv2.imread(input_path)
        if image is None:
            print(f"無法讀取圖片: {file_name}")
            continue

        # 轉為灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        faces = detector(gray)

        # 在圖片上標記人臉
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 儲存結果圖片
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, image)
        # 顯示圖片
        cv2.imshow(f"detected_face {file_name}", image)
        cv2.waitKey(100)
        print(f"已處理並儲存: {output_path}")
        '''
        img = mpimg.imread(output_path)  # 替換成你的圖片路徑
        plt.imshow(img)
        plt.axis('off')  # 隱藏軸
        plt.show()
        '''

# 設定輸入與輸出目錄
input_directory = "./imgs_input"  # 替換為你的圖庫目錄路徑
output_directory = "./imgs_output"  # 替換為輸出結果的目錄

# 執行檢測
detect_and_mark_faces(input_directory, output_directory)
