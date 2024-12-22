
import os
import requests
import tarfile
import shutil
import cv2
import dlib

# 下載並解壓數據集
def download_and_extract_dataset(url, download_path, extract_path):
    """
    下載並解壓數據集
    """
    os.makedirs(extract_path, exist_ok=True)
    print("正在下載數據集...")
    response = requests.get(url, stream=True)
    with open(download_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print("數據集下載完成！")

    print("正在解壓數據集...")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"解壓完成！數據集已存放在: {extract_path}")

    os.remove(download_path)
    print("已刪除壓縮包！")

# 提取圖片到指定目錄
def collect_images(source_dir, target_dir, max_images=100):
    """
    從來源資料夾遞迴提取圖片，並複製到目標資料夾
    """
    os.makedirs(target_dir, exist_ok=True)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    image_count = 0

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, f"image_{image_count+1:04d}.jpg")
                shutil.copy2(source_path, target_path)
                image_count += 1
                if image_count >= max_images:
                    print(f"已成功提取 {max_images} 張圖片到 {target_dir}")
                    return
    print(f"總共找到 {image_count} 張圖片，已全部提取到 {target_dir}")

# 偵測人臉並標記
def detect_and_mark_faces(input_dir, output_dir):
    """
    偵測輸入目錄中的圖片人臉，並在圖片上畫框，結果保存到輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    detector = dlib.get_frontal_face_detector()

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"跳過非圖片檔案: {file_name}")
            continue

        image = cv2.imread(input_path)
        if image is None:
            print(f"無法讀取圖片: {file_name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, image)
        cv2.imshow(f"detected_image {file_name}", image)
        cv2.waitKey(1)
        print(f"已處理並儲存: {output_path}")


def face_predictor(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"跳過非圖片檔案: {file_name}")
            continue

        img = cv2.imread(input_path)
        if img is None:
            print(f"無法讀取圖片: {file_name}")
            continue
        cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  #CNN模型
        
        dets = cnn_detector(img, 1)  #偵測人臉
        print("人臉數：{}".format(len(dets)))
        for i, det in enumerate(dets):
            #det.rect是人臉矩形坐標,det.confidence為信心指數
            face = det.rect
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            print("偵測人臉 {}: 左：{}  上：{}  右：{}  下：{}  信心指數：{}".format(i, left, top, right, bottom, det.confidence))
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)  #畫人臉矩形

        cv2.namedWindow(f"{file_name}", cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f"{file_name}", img)
        cv2.waitKey(1)
        
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, img)
        


# 設定下載參數
dataset_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
download_path = "./lfw-deepfunneled.tgz"
extract_path = "./"

# 下載並解壓數據集
# download_and_extract_dataset(dataset_url, download_path, extract_path)

# 提取圖片
source_directory = "./lfw-deepfunneled"
target_directory = "./imgs_input"
collect_images(source_directory, target_directory, max_images=100)

# 偵測人臉並標記
input_directory = "./imgs_input"
output_directory = "./imgs_output"
detect_and_mark_faces(input_directory, output_directory)

input_directory = './imgs_input_2'
output_directory = './imgs_output_my_result'
detect_and_mark_faces(input_directory, output_directory)
output_directory = './imgs_output_github_result'
face_predictor(input_directory, output_directory)

