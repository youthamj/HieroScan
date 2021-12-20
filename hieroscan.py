from detect import detect
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from collections import Counter, defaultdict
from classify import image_to_gardiner
from preprocess import preprocess
from segmentation import forwardwordsegmentation
from translation import translate
from skimage import io


input1 = io.imread('assets/examples/test0.jpg')
preprocessed = preprocess(input1)
detections = detect(preprocessed)
gardiners, classified_image = image_to_gardiner(preprocessed, detections)
# io.imsave('output0.jpg', classified_image)
segmented_gardiners = forwardwordsegmentation(gardiners)
print(segmented_gardiners)
translated = translate(segmented_gardiners)
print(translated)
