import matplotlib.pyplot as plt
import cv2

def visualize_image_with_bbox(image_path, bboxes, labels):
    """
    Visualize an image with bounding boxes and labels.

    Args:
        image_path (str): Path to the image.
        bboxes (list): List of bounding boxes (x_min, y_min, x_max, y_max).
        labels (list): List of labels corresponding to bounding boxes.
    """
    image = cv2.imread(image_path)
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
