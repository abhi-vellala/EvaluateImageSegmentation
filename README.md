# Evaluation Metrics for Image Segmentation

The evaluation of image segmentation task is similar to a classification problem. We do a truth comparison between ground truth segmentation and predicted segmentation. Here are some of the common evaluation metrics and implementations. Please note that Image segmentation task require tedious evaluation metrics like "Hausdorff distance", "Surface distance", "dice score". Currently, this repo supports some common classification metrics and then build "Dice Score".

Unlike classification problem with numbers, images points towards the pixel values to determine the truth. Here are the definitions of truth valuations with respect to images:

**True Positive:**
The number of pixels that are predicted right with respect to ground truth.

**False Positive:**
The number of pixels that are predicted as a category when there is no mask in ground truth.

**Flase Negative:**
The number of pixels that are not predicted as a category although there is a mask in ground truth maks.

**True Negative:**
The number of pixels that are predicted correctly as empty mask with respect to ground truth mask.

Here are the images to illustrate the above definitions with a 3 x 3 image:

![img.png](img.png)

---

The `evaluate.py` file has the class `EvaluateImageSegmentation` with the methods to determine 

`accuracy` <br/>
`precision` <br/>
`recall` <br/>
`f1score` <br/>
`dice` <br/>

Please note that the f1score and dice score are same. Either one works.






