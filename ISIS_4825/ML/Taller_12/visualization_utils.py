from skimage.segmentation import mark_boundaries

def get_labeled_image(img, label):
    img_mask = mark_boundaries(img, label, outline_color=(0, 1, 0), 
                               color=(1, 0, 0), mode="thick")
    return img_mask