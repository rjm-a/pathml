import pandas as pd
from PIL import Image
from torchvision import transforms

class ImageDataset():
    def __init__(self, 
                 img_path,
                 annotation_file,
                 transform):

        self.img_path = img_path
        self.data = self.load_annotations(annotation_file)
        self.img_transform = transform


    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        path = self.img_path + '/' + img_name
        image = Image.open(path).convert("RGB")
        tensor_image = self.img_transform(image)

        return tensor_image, label


    def __len__(self):
        return(len(self.data))


    def load_annotations(self, annotation_file):
        data = []
        annots = pd.read_csv(annotation_file)

        for index, row in annots.iterrows():
            im_prefix = row['video'][5:7] + '_' + row['video'][7:10]
            im_id = int(row['second']) * 60
            im_name = im_prefix + '/' + row['video'] + '_' + str(im_id) + '.jpg'
            label = row['action_id']

            data.append((im_name, label))

        return data
