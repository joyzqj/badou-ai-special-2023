#最邻近差值算法
import  cv2
import  numpy as np
def resize(img):
    height,width,channels =img.shape
    empty_image=np.zeros((800,800,channels),np.uint8)#构造一个800x800的空图像
    #缩放宽高
    h=800/height
    w=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/h+0.5)
            y=int(j/w+0.5)
            empty_image[i,j]=img[x,y]
    return empty_image

img = cv2.imread("../data/lenna.png")
zoom = resize(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.imwrite("../temp/lenna_nearest.png",zoom)
cv2.waitKey(0)

