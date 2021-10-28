import sys
import copy
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch import nn, utils
# from ../datasets/joint_dataset import JointDataset

sys.path.append('/data/src')
from data_manipulation.data_utils import read_yaml, write_json
from datasets.batch_joint_dataset import BatchJointDataset
from models.base_gru import BaseGRUNet
from utils.random_batch_sampler import RandomBatchSampler
from utils.non_random_batch_sampler import NonRandomBatchSampler
from utils.class_bal_sampler import RandomBatchedWeightedSampler
from utils.transforms import ScaleNoConfJoints
from utils.funcs import get_sample_weights


class TrainInputs():
    def __init__(self, epochs, optimizer, criterion, batch_size, input_size, device):
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = device


def parse_args():
    parser = argparse.ArgumentParser(description='Train a joint lstm model')
    parser.add_argument('--config', help='Provide the target config file')
    parser.add_argument('--name', help='Provide name for model')

    args = parser.parse_args()
    return args


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
            for _, (indices, inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(train_inputs.device)
                labels = labels.to(train_inputs.device)
                # print(f"INPUTS: {inputs}")
                # print(f"LABELS: {labels}")
                # print(f"IDS: {img_ids}")
   
                # tst_lbls = labels[:, -1]
                # print(tst_lbls.shape)         
                # for class_id in range(12):
                    # print('Class {}: {}, {}'.format(class_id, (tst_lbls==class_id).sum().float() / train_inputs.batch_size, (tst_lbls==class_id).sum())) 
               
                
                # if doing batched training init hidden state for each batch
                if phase == 'train':
                    model.init_hidden()
                if phase == 'val': # and labels[0][-4] == -1:
                    model.init_hidden()

                # zero_grad sets the gradients to zero so that they aren't 
                # accumulated on each subsequent backward pass
                # otherwise they won't accurately point towards the min/max
                train_inputs.optimizer.zero_grad()

                # set_grad_enabled(bool) clears intermediate values needed for backpropagation
                # these operations will not require gradients to be computed
                with torch.set_grad_enabled(phase == 'train'):
                    final_outputs = model(inputs)
                    # final_outputs = outputs[:, -1, :]
                    final_labels = labels[:, -1]

                    # print(f"OUTPUTS: {outputs}")
                    # print(f"LAST OUTPUT: {final_outputs}")
                    # print(f"LAST LABEL: {final_labels}")

                    # returns a probability distribution for each label
                    _, predictions = torch.max(final_outputs, dim=1)
                    loss = train_inputs.criterion(final_outputs, final_labels)

                    if phase == 'train':
                        loss.backward()
                        train_inputs.optimizer.step()                
                
                total_loss += loss.item() * inputs.size(0)
                # if torch.sum(predictions == final_labels.data) <= 3:
                #     print("CORRECT")
                #     print(torch.sum(predictions == final_labels.data))
                num_correct += torch.sum(predictions == final_labels.data)

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
            
        torch.save(model.state_dict(), '/data/backup_models/rnn_' + name + '/gru-cb/model_' + str(epoch) +  '.pt')    
        write_json('/data/backup_models/rnn_' + name + '/gru-cb/best_preds.json', [best_preds.cpu().numpy().tolist(), best_pred_labels.cpu().numpy().tolist()])   
        print()

    model.load_state_dict(best_model_wts)
    return model


def main():
    args = parse_args()
    prefix = '/data/src/configs/'
    config = read_yaml(prefix + args.config)

    input_size = 32
    scale_size = (164,290)
    seq_len = 6

    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['layers']
    output_size = config['model']['classes']

    batch_size = config['train_settings']['batch_size']
    num_epochs = config['train_settings']['epochs']
    lr = config['train_settings']['learning_rate']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseGRUNet(input_size, batch_size, hidden_size, num_layers, output_size, device)
    model.cuda()

    ## try using a lr scheduler?
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    joint_transform = ScaleNoConfJoints(size=scale_size)
    train_inputs = TrainInputs(epochs=num_epochs,
                               optimizer=optimizer,
                               criterion=criterion,
                               batch_size=batch_size,
                               input_size=input_size,
                               device=device)

    img_prefix = config['data']['img_prefix']
    train_data = BatchJointDataset(img_prefix=img_prefix,
                                   joints_file=config['data']['train']['joint_file'],
                                   annotation_file=config['data']['train']['ann_file'],
                                   seq_len=seq_len,
                                   data_len=input_size,
                                   batch_size=batch_size,
                                   joint_transform=joint_transform)
    # train_data = JointDataset(img_prefix=img_prefix,
    #                           joints_file=config['train']['joint_file'],
    #                           annotation_file=config['train']['ann_file'],
    #                           seq_len=seq_len,
    #                           scale_size=scale_size)
    # val_data = JointDataset(img_prefix=img_prefix,
    #                         joints_file=config['val']['joint_file'],
    #                         annotation_file=config['val']['ann_file'],
    #                         seq_len=seq_len,
    #                         scale_size=scale_size)
    val_data = BatchJointDataset(img_prefix=img_prefix,
                                 joints_file=config['data']['val']['joint_file'],
                                 annotation_file=config['data']['val']['ann_file'],
                                 seq_len=seq_len,
                                 data_len=input_size,
                                 batch_size=batch_size,
                                 joint_transform=joint_transform)

    # weights, wts_len = get_sample_weights(config['data']['train']['ann_file'])
    # rand_batch_sampler = RandomBatchedWeightedSampler(data_source=train_data, batch_size=batch_size, weights=weights)

    rand_batch_sampler = RandomBatchSampler(data_source=train_data, batch_size=batch_size)
    train_loader = utils.data.DataLoader(train_data, batch_sampler=rand_batch_sampler)
    
    batch_sampler = NonRandomBatchSampler(data_source=val_data, batch_size=batch_size)
    val_loader = utils.data.DataLoader(val_data, batch_sampler=batch_sampler)

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

    # batch sampler repeats indices batch_size number of times
    dataset_sizes = {
        'train' : len(rand_batch_sampler) * batch_size,
        'val' : len(batch_sampler)
    }

    print(model)
    model = train_net(train_inputs, dataloaders, dataset_sizes, model, args.name)
    torch.save(model.state_dict(), '/data/saved_models/rnn/gru_' + args.name +  "_cb.pt")


if __name__ == '__main__':
    main()
