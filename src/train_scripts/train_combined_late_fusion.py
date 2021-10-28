import sys
import copy
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch import nn, utils
from torchvision import models, transforms, datasets

sys.path.append('/data/src')
from data_manipulation.data_utils import read_yaml, write_json
from datasets.batch_joint_img_dataset import BatchJointImgDataset
from models.cnn_gru_late_fusion import LateFusion
from utils.sampler import RandomBatchSampler, NonRandomBatchSampler
from utils.class_bal_sampler import RandomBatchedWeightedSampler
from utils.transforms import ScaleNoConfJoints
from utils.funcs import get_sample_weights


def parse_args():
    parser = argparse.ArgumentParser(description='Train a joint lstm model')
    parser.add_argument('--config', help='Provide the target config file')
    parser.add_argument('--name', help='Provide name for model')

    args = parser.parse_args()
    return args


class ModelParams():
    def __init__(self, input_size, output_size, dr_rate, hidden_size, batch_size, num_layers, cnn_path, device):
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dr_rate
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.cnn_path = cnn_path
        self.device = device


class TrainInputs():
    def __init__(self, epochs, optimizer, criterion, batch_size, device):
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device


def train_net(train_inputs, dataloaders, dataset_sizes, model, name):
    best_accuracy = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_preds = torch.zeros(0, dtype=torch.long).to(train_inputs.device)
    best_pred_labels = torch.zeros(0, dtype=torch.long).to(train_inputs.device)

    for epoch in range(train_inputs.epochs):
        print(f"Epoch {epoch} / {train_inputs.epochs}")
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0
            num_correct = 0
            preds = torch.zeros(0, dtype=torch.long).to(train_inputs.device)
            lbls = torch.zeros(0, dtype=torch.long).to(train_inputs.device)

            dataloader = tqdm(dataloaders[phase])
            for _, (images, inputs, labels) in enumerate(dataloader):
                images = images.to(train_inputs.device)
                inputs = inputs.to(train_inputs.device)
                labels = labels.to(train_inputs.device)
                # print(f"INPUTS: {inputs}")
                # print(f"LABELS: {labels}")
                # print(f"IDS: {img_ids}")
                
                # init hidden state for each batch for batches
                if phase == 'train': #and train_inputs.batch_size > 1:
                    model.init_hidden()

                if phase == 'val': #and labels[0][-4] == -1:
                    model.init_hidden()

                # zero_grad sets the gradients to zero so that they aren't 
                # accumulated on each subsequent backward pass
                # otherwise they won't accurately point towards the min/max
                train_inputs.optimizer.zero_grad()

                # set_grad_enabled(bool) clears intermediate values needed for backpropagation
                # these operations will not require gradients to be computed
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, inputs)

                    # reshape if no dropout
                    # outputs = outputs[:, -1, :]

                    # select targets for only the last timestep of each batch to compute loss
                    labels = labels[:, -1]

                    # print(f"OUTPUTS:")
                    # print(f"LAST OUTPUT: {outputs.shape}")
                    # print(f"LAST LABEL: {labels.shape}")

                    # returns a probability for each label
                    _, predictions = torch.max(outputs, dim=1)
                    loss = train_inputs.criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        train_inputs.optimizer.step()                
                
                total_loss += loss.item() * inputs.size(0)
                num_correct += torch.sum(predictions == labels.data)

                if phase == 'val':
                    preds = torch.cat([preds, predictions])
                    lbls = torch.cat([lbls, final_labels])

            epoch_loss = total_loss / dataset_sizes[phase]
            epoch_accuracy = num_correct.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss} Accuracy: {epoch_accuracy}')

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                best_preds = preds
                best_pred_labels = lbls
            
        torch.save(model.state_dict(), '/data/backup_models/cnn_rnn_' + name + '/combined-late-v1/model_' + str(epoch) +  '.pt')           
        write_json('/data/backup_models/cnn_rnn_' + name + '/combined-late-v1/best_preds.json', [best_preds.cpu().numpy().tolist(), best_pred_labels.cpu().numpy().tolist()])
        print()

    model.load_state_dict(best_model_wts)
    return model


def main():
    args = parse_args()
    prefix = '/data/src/configs/'
    config = read_yaml(prefix + args.config)

    input_size = 32
    seq_len = 6

    height = config['images']['height']
    width = config['images']['width']
    cnn_path = config['images']['cnn_path']

    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['layers']
    output_size = config['model']['classes']
    
    batch_size = config['train_settings']['batch_size']
    num_epochs = config['train_settings']['epochs']
    lr = config['train_settings']['learning_rate']
    dr_rate = config['train_settings']['dropout']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_params = ModelParams(input_size=input_size,
                               output_size=output_size,
                               dr_rate=dr_rate,
                               hidden_size=hidden_size,
                               batch_size=batch_size,
                               num_layers=num_layers,
                               cnn_path=cnn_path,
                               device=device)

    model = LateFusion(model_params)
    model.cuda()

    ## need a scheduler to decrease learning rate over time if decide to use SGD
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    joint_transform = ScaleNoConfJoints(size=(height,width))
    train_inputs = TrainInputs(epochs=num_epochs,
                               optimizer=optimizer,
                               criterion=criterion,
                               batch_size=batch_size,
                               device=device)

    normalizer = transforms.Normalize(mean=[0.3384, 0.3171, 0.3217], std=[0.4073, 0.3909, 0.3927])
    transformer = transforms.Compose([
            transforms.Resize(size=(height,width)),
            transforms.ToTensor(),
            normalizer
    ])

    train_data = BatchJointImgDataset(img_prefix=config['data']['img_prefix'],
                                      tensor_img_prefix=config['data']['tensor_img_prefix'],
                                      joints_file=config['data']['train']['joint_file'],
                                      annotation_file=config['data']['train']['ann_file'],
                                      seq_len=seq_len,
                                      joint_len=input_size,
                                      height=height,
                                      width=width,
                                      im_transform=transformer,
                                      joint_transform=joint_transform)
    # train_data = JointImgDataset(img_prefix=config['data']['img_prefix'],
    #                              joints_file=config['data']['train']['joint_file'],
    #                              annotation_file=config['data']['train']['ann_file'],
    #                              seq_len=seq_len,
    #                              joint_len=input_size,
    #                              height=height,
    #                              width=width,
    #                              im_transform=transformer)
    # val_data = JointImgDataset(img_prefix=config['data']['img_prefix'],
    #                            joints_file=config['data']['val']['joint_file'],
    #                            annotation_file=config['data']['val']['ann_file'],
    #                            seq_len=seq_len,
    #                            joint_len=input_size,
    #                            height=height,
    #                            width=width,
    #                            im_transform=transformer)
    val_data = BatchJointImgDataset(img_prefix=config['data']['img_prefix'],
                                    tensor_img_prefix=config['data']['tensor_img_prefix'],
                                    joints_file=config['data']['val']['joint_file'],
                                    annotation_file=config['data']['val']['ann_file'],
                                    seq_len=seq_len,
                                    joint_len=input_size,
                                    height=height,
                                    width=width,
                                    im_transform=transformer,
                                    joint_transform=joint_transform)

    weights, wts_len = get_sample_weights(config['data']['train']['ann_file'])
    # rand_batch_sampler = RandomBatchedWeightedSampler(data_source=train_data, batch_size=batch_size, seq_len=seq_len, weights=weights)

    rand_batch_sampler = NonRandomBatchSampler(data_source=train_data, batch_size=batch_size, seq_len=seq_len)
    # rand_batch_sampler = RandomBatchSampler(data_source=train_data, batch_size=batch_size, seq_len=seq_len)
    train_loader = utils.data.DataLoader(train_data, batch_sampler=rand_batch_sampler, num_workers=10)
    
    nrand_batch_sampler = NonRandomBatchSampler(data_source=val_data, batch_size=batch_size, seq_len=seq_len)
    val_loader = utils.data.DataLoader(val_data, batch_sampler=nrand_batch_sampler, num_workers=4)

    # train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    # val_loader = utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }

    # dataset_sizes = {
    #     'train' : len(train_data),
    #     'val' : len(val_data)
    # }

    dataset_sizes = {
        'train' : len(rand_batch_sampler) * batch_size,
        'val' : len(val_data)
    }

    # print(model)
    model = train_net(train_inputs, dataloaders, dataset_sizes, model, args.name)
    torch.save(model.state_dict(), '/data/saved_models/cnn-rnn/combined_' + args.name +  ".pt")


if __name__ == '__main__':
    main()

