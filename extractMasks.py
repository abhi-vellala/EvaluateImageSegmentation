from PIL import Image
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt



def extract_mask(img, annotation_path, save_path, mask_name):
    MASK_WIDTH = np.array(img).shape[0]
    MASK_HEIGHT = np.array(img).shape[1]
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))

    with open(annotation_path) as f:
        data = json.load(f)

    for i in data:
        name = i
    x_points = data[name]["regions"][0]["shape_attributes"]["all_points_x"]
    y_points = data[name]["regions"][0]["shape_attributes"]["all_points_y"]
    mask_points = {}
    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])

    mask_points["image"] = all_points
    arr = np.array(mask_points["image"])
    cv2.fillPoly(mask, [arr], color=(255))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.imshow(mask, cmap='copper', alpha=0.5)
    fig.savefig(f"{save_path}/{mask_name}_overlay.png")
    plt.close()

    cv2.imwrite(f"{save_path}/{mask_name}.png", mask)

    print("Success")


if __name__ == "__main__":
    img = Image.open("./data/just_cat.jpeg")
    annotation_path = "./data/gt_annotation.json"
    save_path = "./data/"
    mask_name = "gt_mask"
    extract_mask(img, annotation_path, save_path, mask_name)



