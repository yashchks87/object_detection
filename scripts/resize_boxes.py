def fix_original_values(box):
    updated_box = [box[0], box[1], box[2]- box[0], box[3]- box[1]]
    return updated_box

def resize_values(box, old_img_size, target_img_size, return_old_box = True, is_torch = False):
    if is_torch:
        width, height = old_img_size[2], old_img_size[1]
    else:
        width, height = old_img_size[1], old_img_size[0]
    old_updated_box = fix_original_values(box)
    x1, y1, x2, y2 = old_updated_box[0], old_updated_box[1], old_updated_box[2], old_updated_box[3]
    x1_o = x1 / width
    y1_o = y1 / height
    x2_o = x2 / width
    y2_o = y2 / height

    x1_n = target_img_size[0] * x1_o
    y1_n = target_img_size[1] * y1_o
    x2_n = target_img_size[0] * x2_o
    y2_n = target_img_size[1] * y2_o

    if return_old_box:
        return [x1_n, y1_n, x2_n, y2_n], [x1, y1, x2, y2]
    return [x1_n, y1_n, x2_n, y2_n]