from evaluate import EvaluateImageSegmentation
from PIL import Image
import numpy as np
# testing changes

img = Image.open("./data/images/just_cat.jpeg")
gt_mask = np.array(Image.open("./data/images/gt_mask.png"))
pred_mask = np.array(Image.open("./data/images/pred_mask.png"))

evaluate = EvaluateImageSegmentation(gt_mask, pred_mask)

print(f"Accuracy: {round(evaluate.accuracy(),4)}")
print(f"Precision: {round(evaluate.precision(),4)}")
print(f"Recall: {round(evaluate.recall(),4)}")
print(f"F1 Score: {round(evaluate.f1score(), 4)}")
print(f"Dice: {round(evaluate.dice(), 4)}")
print(f"Intersection over Union(IoU): {round(evaluate.IoU(), 4)}")
print(f"Hausdorff Distance: {round(evaluate.hausdorff_distance(),4)}")
print("Confusion Matrix:")
evaluate.get_confusion_matrix()