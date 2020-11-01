from skimage.segmentation import mark_boundaries

def get_labeled_image(img, label, outline_color=(1, 0, 0), 
                        color=(1, 0, 0)):
    img_mask = mark_boundaries(img, label, outline_color=outline_color, 
                               color=color, mode="thick")
    return img_mask