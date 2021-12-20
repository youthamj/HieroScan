from torchvision.utils import draw_bounding_boxes
import torch
from skimage import io
import pandas as pd

detection_model = None


def load_model():
    global detection_model
    with open("assets\detect_assets\model.ts", "rb") as f:
        detection_model = torch.jit.load(f)
    print('Detection model loaded successfully.')


def detect(img):
    tr_img = img.transpose((-1, 0, 1))
    inputs = ({"image": torch.tensor(tr_img)},)
    global detection_model
    detection_model = torch.jit.script(detection_model)
    with torch.no_grad():
        inference = detection_model(inputs)
    result_img = draw_bounding_boxes(torch.tensor(tr_img), inference[0]['pred_boxes'], colors='red', width=2)
    # save_image(F.to_pil_image(result_tensor), 'result.png', format='png')
    # cv2.imwrite('result.png', F.to_pil_image(result_tensor))
    # res_image = F.to_pil_image(result_tensor)
    # res_image.save('result.png')
    result_img = result_img.numpy().transpose((1, 2, 0))
    # cv2.imwrite('result.png', )
    io.imsave('result.png', result_img)
    detections = {'bboxes': inference[0]['pred_boxes'].numpy().tolist()}
    detected_df = pd.DataFrame(detections, columns=['bboxes'])
    coordinates = detected_df['bboxes']
    return coordinates


load_model()
if __name__ == "__main__":
    img1 = io.imread('results/preprocess0.jpg')
    res = detect(img1)
    print(res['bboxes'])
    # input = open("input.txt", "w")
    # for bbox in res['bboxes']:
    #     input.write(sentence + "\n")