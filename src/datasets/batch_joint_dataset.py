import sys
import numpy as np
import pandas as pd
import os.path as osp
import torch
from torch.utils.data import Dataset

sys.path.append('/data/src')
from data_manipulation.data_utils import read_json, image_dims, read_yaml
# from utils.transforms import ScaleNoConfJoints

class BatchJointDataset(Dataset):
    ##########################################################################
    ######  Returns tensor of joint locations and label for each image  ######
    ######  in batches of length seq_len where each seq is a time step  ######
    ##########################################################################

    def __init__(self, 
                 img_prefix,
                 joints_file,
                 annotation_file,
                 seq_len,
                 data_len,
                 batch_size,
                 joint_transform):

        self.img_prefix = img_prefix
        self.joints_file = joints_file
        self.annotation_file = annotation_file
        self.seq_len = seq_len
        self.data_len = data_len
        self.video_metadata = {}

        self.joint_transform = joint_transform
        joint_records = self.load_annotations(img_prefix, annotation_file, joints_file)
        self.joint_records, self.labels = self.transform_records(joint_records)


    def __len__(self):
        return len(self.joint_records)


    def __getitem__(self, idx):
        if idx in self.video_metadata.keys():
            start = self.video_metadata[idx] 
            diff = idx - start + 1

            record = torch.Tensor(self.joint_records[start : idx + 1])
            label = self.labels[start : idx + 1]

            prev_records = torch.from_numpy(np.full(shape=(self.seq_len - diff, self.data_len), fill_value=-1).astype(np.float32))
            prev_labels = torch.from_numpy(np.full(shape=self.seq_len - diff, fill_value=-1))

            # print()
            # print(prev_records.shape)
            # print(record.shape)
            # print()
            return idx, torch.cat((prev_records, record)), torch.cat((prev_labels, label))

        # print(self.joint_records.shape)
        # print(self.labels.shape)
        # print(f"INDEX: {idx}")
        # print(f"FROM: {idx - self.seq_len + 1}")
        # print(f"TO: {idx + 1}")
        # print(self.joint_records[idx - self.seq_len + 1 : idx + 1].shape)
        return idx, self.joint_records[idx - self.seq_len + 1 : idx + 1], self.labels[idx - self.seq_len + 1 : idx + 1]


    ## remove previous records and labels for the start of a new video
    def reset_history(self):
        self.prev_records = torch.from_numpy(np.full(shape=(self.seq_len, self.data_len), fill_value=-1).astype(np.float32))
        self.prev_labels = torch.from_numpy(np.full(shape=self.seq_len, fill_value=-1))


    def transform_records(self, records):
        joints = []
        labels = []

        last_vid = None

        for index, data in enumerate(records):
            if last_vid != data['image_id'][:10]:
                last_vid = data['image_id'][:10]
                # print(index)
                # print(data['image_id'])
                self.video_metadata.update({invalid_index : index for invalid_index in range(index, index + (self.seq_len - 1))})
                
            if data['width'] == -1:
                # no joints were found for this frame
                new_joints = torch.from_numpy(data['joints'])

            else:
                # scale joints as if image is being resized
                new_joints = self.joint_transform(data['height'], data['width'], data['joints'])
                new_joints = torch.FloatTensor(new_joints)

            joints.append(new_joints)
            # labels.append(torch.LongTensor([data['action_id']]))
            labels.append(data['action_id'])

        return torch.stack(joints), torch.LongTensor(labels)


    ## holds all annotations in memory 
    ## if this gets too big can store indices to json arrays
    def load_annotations(self, img_prefix, annotation_file, joints_file):
        # Create an array of dicts
        # Each dict contains image name, action id, tensor of joints
        joint_records = []

        annots = pd.read_csv(annotation_file)
        joints = read_json(joints_file)

        joints_index = 0
        for index, row in annots.iterrows():
            joints_index, record = self.joints_record(row, joints_index, joints, img_prefix)
            joint_records.append(record)

        return joint_records


    ## create a joint record for each annotation
    def joints_record(self, annot_row, joints_index, joints, img_prefix):
        annot_row_vid = annot_row['video']
        img_id = annot_row_vid + '_' + str(annot_row['second'] * 60) + '.jpg'
        img_folder = annot_row_vid[5:7] + '_' + annot_row_vid[7:]

        empty_record = {'image_id': img_id,
                        'action_id': annot_row['action_id'],
                        'joints': np.full(shape=self.data_len, fill_value=-1, dtype=np.float32),
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
            
        while joints_index < len(joints) and joints[joints_index]['image_id'] == img_id:
            # use the skeleton that is the most visible 
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
        
