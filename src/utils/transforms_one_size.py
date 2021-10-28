class ScaleJoints(object):
    def __init__(self, size):
        self.size = size


    def __call__(self, height, width, points):
        new_points = []
        scale = self.get_new_img_scale(height, width)

        for i in range(0, len(points), 3):
            x = points[i] / scale
            y = points[i + 1] / scale

            new_points += [x, y, points[i + 2]]

        return new_points


    def get_new_img_scale(self, height, width):
        if width >= height:
            scale = height / self.size

        elif width < height:
            scale = width / self.size

        return scale


class ScaleImgJoints(object):
    ## Adjust y points after scaling to account for image crop
    def __init__(self, size, crop_width):
        self.size = size
        self.crop_width = crop_width


    def __call__(self, height, width, points):
        new_points = []
        scale = self.get_new_img_scale(height, width)
        scaled_width = width / scale
        diff = scaled_width - self.crop_width

        for i in range(0, len(points), 3):
            x = points[i] // scale
            y = points[i + 1] // scale

            # account for center crop by moving y location
            if diff >= 2:
                y = y - (diff // 2)

            new_points += [x, y, points[i + 2]]

        return new_points


    def get_new_img_scale(self, height, width):
        if width >= height:
            scale = height / self.size

        elif width < height:
            scale = width / self.size

        return scale
