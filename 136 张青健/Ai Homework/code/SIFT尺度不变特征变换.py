# SIFT特征提取和匹配具体步骤
# 1. 生成高斯差分金字塔（DOG金字塔），尺度空间构建
# 2. 空间极值点检测（关键点的初步查探）
# 3. 稳定关键点的精确定位
# 4. 稳定关键点方向信息分配
# 5. 关键点描述
# 6. 特征点匹配
import cv2
import numpy as np

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    # 获取图像1和图像2的高度和宽度
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    # 创建一个空白图像，用于显示两个图像并绘制匹配线
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    # 获取匹配点的索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 获取匹配点的坐标，并将图像2的坐标偏移w1个单位，以便在vis上正确绘制匹配线
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    # 在vis上绘制匹配线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    # 创建一个窗口并显示绘制了匹配线的图像
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)

# 读取图像1和图像2
img1_gray = cv2.imread("../data/1.jpg")
img2_gray = cv2.imread("../data/2.jpg")

# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 在图像1和图像2上检测关键点和计算描述符
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 使用Brute-Force匹配器进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# 选择好的匹配点
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

# 绘制前20个匹配点的匹配线
sift_img=drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])
cv2.imwrite("../temp/sift.jpg",sift_img)
# 等待按键响应，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
