from torchvision.utils import draw_bounding_boxes
import torch
# import cv2
from skimage import io

# from PIL import Image

model = None


def load_model():
    global model
    with open("assets\detect_assets\model.ts", "rb") as f:
        model = torch.jit.load(f)
    print('Detection model loaded successfully.')


def predict(img):
    tr_img = img.transpose((-1, 0, 1))
    inputs = ({"image": torch.tensor(tr_img)},)
    global model
    model = torch.jit.script(model)
    with torch.no_grad():
        inference = model(inputs)
    result_img = draw_bounding_boxes(torch.tensor(tr_img), inference[0]['pred_boxes'], colors='red', width=2)
    # save_image(F.to_pil_image(result_tensor), 'result.png', format='png')
    # cv2.imwrite('result.png', F.to_pil_image(result_tensor))
    # res_image = F.to_pil_image(result_tensor)
    # res_image.save('result.png')
    result_img = result_img.numpy().transpose((1, 2, 0))
    # cv2.imwrite('result.png', )
    io.imsave('result.png', result_img)
    res_obj = {'bboxes': inference[0]['pred_boxes'].numpy(), 'image': result_img}
    return res_obj


load_model()
if __name__ == "__main__":
    img1 = io.imread('assets/examples/0.jpg')
    img2 = io.imread('assets/examples/51.jpg')
    img3 = io.imread('assets/examples/148.jpg')
    img4 = io.imread('assets/examples/199.jpg')
    res = predict(img1)
    print(res['bboxes'])
