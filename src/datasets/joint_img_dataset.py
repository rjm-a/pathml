import sys
import numpy as np
import pandas as pd
import os.path as osp
import torch
from torch.utils.data import Dataset
from PIL import Image

sys.path.append('/data/src')
from data_manipulation.data_utils import read_json, image_dims, read_yaml
from utils.transforms import ScaleImgJoints


class JointImgDataset(Dataset):
    ##########################################################################
    ######  Returns tensor of joint locations and label for each image  ######
    ######  in batches of length seq_len where each seq is a time step  ######
    ######  use for testing - loads frame by frame with history reset   ######
    ##########################################################################

    def __init__(self, 
                 img_prefix,
                 joints_file,
                 annotation_file,
                 seq_len,
                 joint_len,
                 height,
                 width,
                 im_transform):

        self.img_prefix = img_prefix
        self.joints_file = joints_file
        self.annotation_file = annotation_file
        self.seq_len = seq_len
        self.joint_len = joint_len
        self.height = height
        self.width = width
        self.im_transform = im_transform
        # self.video_boundaries = []

        self.joint_records = self.load_annotations(img_prefix, annotation_file, joints_file)
        self.joint_transform = ScaleImgJoints(size=height, crop_width=width)
        self.im_transform = im_transform

        self.last_id = None
        self.reset_history()


    def __len__(self):
        return len(self.joint_records)


    def __getitem__(self, idx):
        # get the keypoints for the current video frame
        data = self.joint_records[idx]
        if data['joints'][0] == -1:
            # no joints were found for this frame
            new_joints = torch.from_numpy(data['joints'])

        else:
            # scale joints as if image is being resized
            new_joints = self.joint_transform(data['height'], data['width'], data['joints'])
            new_joints = torch.FloatTensor(new_joints)


        ## this is only necessary if batch size = 1
        ## prevents predicting start of new video using images from last video
        if data['image_id'][:10] != self.last_id:
            self.reset_history()
            self.last_id = data['image_id'][:10]

        # add image to previous images
        img_id = data['image_id']
        img_folder = img_id[5:7] + '_' + img_id[7:10]
        image = Image.open(self.img_prefix + '/' + img_folder + '/' + img_id).convert("RGB")
        tensor_image = self.im_transform(image)
        self.prev_images = torch.cat((self.prev_images[1:], tensor_image.unsqueeze(0)))

        # add scaled joints and new label to time series
        self.prev_records = torch.cat((self.prev_records[1:], new_joints.unsqueeze(0)))
        self.prev_labels = torch.cat((self.prev_labels[1:], torch.LongTensor([data['action_id']])))

        return self.prev_images, self.prev_records, self.prev_labels


    ## remove previous records and labels for the start of a new video
    def reset_history(self):
        self.prev_images = torch.zeros(self.seq_len, 3, self.height, self.width)
        self.prev_records = torch.from_numpy(np.full(shape=(self.seq_len, self.joint_len), fill_value=-1).astype(np.float32))
        self.prev_labels = torch.from_numpy(np.full(shape=self.seq_len, fill_value=-1))


    ## holds all annotations in memory 
    ## if this gets too big can store indices to json arrays
    def load_annotations(self, img_prefix, annotation_file, joints_file):
        # Create an array of dicts
        # Each dict contains image name, action id, tensor of joints
        joint_records = []

        annots = pd.read_csv(annotation_file)
        joints = read_json(joints_file)
        # last_vid = None

        joints_index = 0
        for index, row in annots.iterrows():
            joints_index, record = self.joints_record(row, joints_index, joints, img_prefix)
            joint_records.append(record)

            # found the start of a new video - record starting index of new video
            # if last_vid != row['video']:
            #     last_vid = row['video']
            #     self.video_boundaries.append(index)

        return joint_records


    ## create a joint record for each annotation
    def joints_record(self, annot_row, joints_index, joints, img_prefix):
        annot_row_vid = annot_row['video']
        img_id = annot_row_vid + '_' + str(annot_row['second'] * 60) + '.jpg'
        img_folder = annot_row_vid[5:7] + '_' + annot_row_vid[7:]

        empty_record = {'image_id': img_id,
                        'action_id': annot_row['action_id'],
                        'joints': np.full(shape=self.joint_len, fill_value=-1, dtype=np.float32),
                        'width': -1,
                        'height': -1}

        # no skeleton was found (previous video - need to skip to next video)
        if annot_row_vid not in joints[joints_index]['image_id']:
            # print(f"NONE: {annot_row_vid} {annot_row['second'] * 60} - {joints[joints_index]['image_id']}")

            # current skeleton is from the next video - for this label return a "no skeleton found" record
            if int(joints[joints_index]['image_id'].split('_')[2].strip('.jpg')) / 60 < annot_row['second']:
                return joints_index, empty_record

            # for some reason there are skeleton records that extend past the end of the annotations for the current video
            while annot_row_vid not in joints[joints_index]['image_id'] and int(joints[joints_index]['image_id'].split('_')[2].strip('.jpg')) / 60 > annot_row['second']:
                joints_index += 1

        # no skeleton was found and time-wise the skeleton records are past where the annotations are
        if int(joints[joints_index]['image_id'].split('_')[2].strip('.jpg')) / 60 > annot_row['second']:
            # print(f"NONE: {annot_row_vid} {annot_row['second'] * 60} - {joints[joints_index]['image_id']}")
            return joints_index, empty_record


        final_entry = None
        prev = -1
        # if multiple identifications were made, choose one skeleton
        # print()
        # print(joints[joints_index]['image_id'])
        # print(img_id)
        # print()
        while joints_index < len(joints) and joints[joints_index]['image_id'] == img_id:
            # use the skeleton that is the most visible 

            # print()
            # print(joints[joints_index]['image_id'])
            # print(img_id)
            # print()

            nonzero = np.count_nonzero(joints[joints_index]['keypoints'])
            if nonzero > prev:
                prev = nonzero
                final_entry = joints[joints_index]
            joints_index += 1

        try:
            assert final_entry is not None
        except:
            print(f'joints: {joints[joints_index]["image_id"]}')
            print(f'annotation: {img_id}')

        # width and height needed for resizing the skeleton (scaling the image)
        width, height = image_dims(img_prefix + '/' + img_folder + '/' + img_id)
        record = {'image_id': img_id,
                  'action_id': annot_row['action_id'],
                  'joints': final_entry['keypoints'],
                  'width': width,
                  'height': height}

        return joints_index, record
        
