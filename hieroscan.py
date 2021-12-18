from skimage import io

from detect import predict

img1 = io.imread('assets/examples/0.jpg')
predict(img1)
