from detect import detect
from classify import image_to_gardiner
from visualize import visualize
from preprocess import preprocess
from segmentation import forwardwordsegmentation
from translation import translate
from skimage import io

input1 = io.imread('assets/examples/test0.jpg')
preprocessed = preprocess(input1)
detections = detect(preprocessed)
classified_img, sentences = image_to_gardiner(preprocessed, detections)
io.imsave('classification0.png', classified_img)

