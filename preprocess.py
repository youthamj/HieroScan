from skimage import io
import cv2
import numpy as np
from skimage.exposure import adjust_sigmoid


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new = adjust_sigmoid(gray, cutoff=0.6, gain=5, inv=False)
    stacked_img = np.stack((new,) * 3, axis=-1)
    return stacked_img


##----------main test----
if __name__ == "__main__":
    img = io.imread('assets/examples/73.jpg')
    preprocessed = preprocess(img)
    # plt.imshow(preprocessed)
    cv2.imwrite('results/preprocess73.jpg', preprocessed)