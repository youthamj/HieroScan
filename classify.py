import os
import pandas as pd
import cv2
import numpy as np
from skimage import io, transform, color
from visualize import visualize

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import pickle
from sklearn.utils import shuffle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

classification_model = None
multi_anchor_img = None
multi_anchor_label = None
language_model = None
def crop_vertical(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray, sigma=3, low_threshold=10, high_threshold=50)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(edges, theta=tested_angles)
    _, angles, dist = hough_line_peaks(h, theta, d, 25)
    dist = np.append(dist, [0, gray.shape[1]])
    sub_images = []
    dist = sorted(dist)
    for i in range(len(dist) - 1):
        sub_images.append(gray[:, int(dist[i]):int(dist[i + 1])])
    return sub_images, dist


# ----------------------------------------------------
# -------------------------function changed---------------
# -------------------------------------------------------
def move_coord(coordinates, dist):
    new_coordinates = []
    for d in range(len(dist) - 1):
        new_coordinates.append([])

    for cor in coordinates:
        for i in range(len(dist) - 1):
            if cor[0] >= dist[i] and cor[0] <= dist[i + 1]:
                new_cor = cor.copy()
                new_cor[0] = new_cor[0] - dist[i]
                new_coordinates[i].append(new_cor)
    ordered_coor = []
    for coor in new_coordinates:
        new_coordinates_df = pd.DataFrame(coor, columns=['X', 'Y', 'L', 'W'])
        new_coordinates_df = new_coordinates_df.sort_values(by=['Y', 'X'])
        new_coordinates_df['order'] = np.arange(new_coordinates_df.shape[0], dtype=int)
        ordered_coordinates = new_coordinates_df.to_numpy()
        ordered_coor.append(ordered_coordinates)
    ordered_full_coordinates = []
    for i in range(len(ordered_coor)):
        ordered_full_coordinate = []
        for j in range(len(ordered_coor[i])):
            temp = ordered_coor[i][j].copy()
            temp[0] = temp[0] + dist[i]
            ordered_full_coordinate.append(temp)
        ordered_full_coordinates.append(ordered_full_coordinate)
    return ordered_coor, ordered_full_coordinates
    # -----------------------------------------


def get_glyphs(gray_image, coordinates):
    glyphs = []
    for coordinate in coordinates:
        cor = np.asarray(coordinate).astype(int)
        cor[2] = cor[2] - cor[0]
        cor[3] = cor[3] - cor[1]
        glyph_img = gray_image[cor[1]:cor[1] + cor[3], cor[0]:cor[0] + cor[2]]
        glyphs.append(glyph_img)
    return glyphs


def add_padding(img, new_shape=(75, 50)):
    h, w = img.shape
    exp_h, exp_w = new_shape
    addedh = 0
    addedw = 0

    if w <= exp_w and h <= exp_h:
        addedh = exp_h - h
        addedw = exp_w - w
    else:
        h0 = h * exp_w
        w0 = w * exp_h
        if h0 >= w0:
            addedw = (h0 - w0) / exp_h
        else:
            addedh = (w0 - h0) / exp_w

    addedh = int(addedh / 2)
    addedw = int(addedw / 2)

    bordered = cv2.copyMakeBorder(img, addedh, addedh, addedw, addedw, cv2.BORDER_REPLICATE)

    bordered = cv2.GaussianBlur(bordered, (15, 15), 0)
    bordered[addedh:addedh + h, addedw:addedw + w] = img
    resized = cv2.resize(bordered, (exp_w, exp_h))
    return resized


def pad_images(sentence_images):
    padded_images = []
    for img in sentence_images:
        padded_images.append(add_padding(img))
    return padded_images


def lm_next(model, prev):
    pred = dict(eval('model' + str(prev)))
    next_scores = sorted(pred.items(), key=lambda item: item[1], reverse=True)
    out = dict(next_scores)
    if len(list(out.keys())) == 0:
        out = {'None': 0}
    return out


def whichGlyph_pair(image, anchor_img, anchor_label):
    N, w, h, _ = anchor_img.shape

    test_image = np.asarray([image] * N).reshape(N, w, h, 1)

    anchor_label, test_image, anchor_img = shuffle(anchor_label, test_image, anchor_img)


    return test_image, anchor_img, anchor_label


def whichGlyph(model, image, anchor_img, anchor_label):
    test_image, anchor_img, targets = whichGlyph_pair(image, anchor_img, anchor_label)
    probs = model.predict([test_image, anchor_img])
    return probs, anchor_img, targets


def predict_multi_anchor(image, multi_anchor_img, multi_anchor_label, model):
    multi_N = multi_anchor_img.shape[0]
    final_scores = np.zeros((multi_anchor_label[0].shape[0], 1))
    for j in range(multi_N):
        predicted, anchor_imgs, targets = whichGlyph(model, image, multi_anchor_img[j], multi_anchor_label[j])
        zipped_lists = zip(targets, predicted)
        sorted_pairs = sorted(zipped_lists)
        tuples = zip(*sorted_pairs)
        targets, predicted = [list(tuple) for tuple in tuples]
        final_scores = np.asarray(final_scores) + np.asarray(predicted)
    return final_scores, np.asarray(targets)


def choose_next(clf_3, score_3, language_model, prev):
    freq = lm_next(language_model, prev)
    freq_3 = []
    for pred in clf_3:
        if pred in freq.keys():
            freq_3.append(freq[pred])
        else:
            freq_3.append(0)
    freq_3 = np.array(freq_3)
    score_3 = np.array(score_3).flatten()
    score_3_sum = score_3 / np.sum(score_3)
    freq_3_exp = np.exp(freq_3) / sum(np.exp(freq_3))
    scores = score_3_sum + 2 * freq_3_exp
    predicted = clf_3[np.argmax(scores)]
    return predicted


def predict_lm(Xtest, multi_anchor_img, multi_anchor_label, model, language_model):
    preds = []
    for new in Xtest:
        if len(preds) < 2:
            predicted, targets = predict_multi_anchor(new, multi_anchor_img, multi_anchor_label, model)
            sort_index = np.argsort(np.asarray(predicted).reshape(len(predicted), ))
            targ = targets[sort_index[-1]]

            if targ == 'UNKNOWN':
                targ = targets[sort_index[-2]]
            preds.append(targ)
        else:
            predicted, targets = predict_multi_anchor(new, multi_anchor_img, multi_anchor_label, model)
            sort_index = np.argsort(np.asarray(predicted).reshape(len(predicted), ))
            predicted = choose_next(targets[sort_index[-3:]], predicted[sort_index[-3:]], language_model, preds[-2:])
            preds.append(predicted)
    return preds


def predict_all(glyphs, model, multi_anchor_img, multi_anchor_label, language_model, coordinates):
    predictions = []
    pred_arr = []

    for glyph_list in glyphs:
        sentence_padded = pad_images(glyph_list)
        preds = predict_lm(sentence_padded, multi_anchor_img, multi_anchor_label, model, language_model)
        pred_sub = "".join(preds)
        predictions.append(pred_sub)
        pred_arr.append(preds)
    return predictions, pred_arr


def image_to_gardiner(img, coordinates):
    global multi_anchor_img, multi_anchor_label, classification_model, language_model

    sub_images, dist = crop_vertical(img)

    new_coordinates, ordered_full_coordinates = move_coord(coordinates, dist)

    glyphs = []
    for i in range(len(sub_images)):
        glyphs.append(get_glyphs(sub_images[i], new_coordinates[i]))

    preds, pred_arr = predict_all(glyphs, classification_model, multi_anchor_img, multi_anchor_label, language_model, coordinates)

    final_coor = []
    for i in range(len(ordered_full_coordinates)):
        cor_list = []
        for j in range(len(ordered_full_coordinates[i])):
            # print(i,j)
            cor_list.append({"coor": ordered_full_coordinates[i][j], "pred": [pred_arr[i][j]]})
        final_coor.append(cor_list)
    classified_img = visualize(img, final_coor)
    return preds, classified_img
    # -----------------------------------------


# read all files needed
# with open("assets/classify_assets/fine_tuned_model(1)_new.pickle", "rb") as f:
#     (model) = pickle.load(f)

# print(model.summary())
# with tf.device('/cpu:0'):
def load_models():
    global pickle, classification_model, multi_anchor_img, multi_anchor_label, language_model

    classification_model = keras.models.load_model("assets/classify_assets/final_model.h5")

    with open("assets/classify_assets/multi_anchor.pickle", "rb") as f:
        (multi_anchor_img, multi_anchor_label) = pickle.load(f)

    import dill as pickle

    with open("assets/classify_assets/language_model_sent.pkl", "rb") as f:
        language_model = pickle.load(f)
    print('Classify models loaded successfully.')

load_models()
if __name__ == "__main__":
    from detect import detect

    test_img = io.imread('results/preprocess73.jpg')
    detections = detect(test_img)

    final_preds, image = image_to_gardiner(test_img, detections)
    # , multi_anchor_img, multi_anchor_label, model, language_model
    print(final_preds)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    # io.imsave('classification0.png', image)
