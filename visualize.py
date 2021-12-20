import cv2

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def _visualize_bbox(img, bbox, color=BOX_COLOR, thickness=1):
    """Visualizes a single bounding box on the image"""
    # print(bbox['coor'])
    x_min, y_min, x_max, y_max, num = bbox['coor']
    class_name = str(int(num)) + ":" + bbox['pred'][0]
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, coor_pred):
    img = image.copy()
    for bboxes in coor_pred:
        for bbox in bboxes:
            # class_name = category_id_to_name[category_id]
            img = _visualize_bbox(img, bbox)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    return img
