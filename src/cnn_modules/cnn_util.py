import sys
import copy
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.append('/data/src')
from data_manipulation.data_utils import write_json

class TrainInputs():
  def __init__(self, num_epochs, optimizer, criterion, scheduler, dataloaders, dataset_sizes, backup_path):
    self.num_epochs = num_epochs
    self.optimizer = optimizer
    self.criterion = criterion
    self.scheduler = scheduler
    self.dataloaders = dataloaders
    self.dataset_sizes = dataset_sizes
    self.backup_path = backup_path


def trainModel(train_inputs, model, device):
  best_accuracy = 0
  best_model_wts = copy.deepcopy(model.state_dict())
  best_preds = torch.zeros(0, dtype=torch.long).to(device)
  best_pred_labels = torch.zeros(0, dtype=torch.long).to(device)

  for epoch in range(train_inputs.num_epochs + 1):
    print(f"Epoch {epoch} / {train_inputs.num_epochs}")
    print('-------------')

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
      
      total_loss = 0
      num_correct = 0
      preds = torch.zeros(0, dtype=torch.long).to(device)
      lbls = torch.zeros(0, dtype=torch.long).to(device)

      loader = tqdm(train_inputs.dataloaders[phase])
      for _, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        train_inputs.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, predictions = torch.max(outputs, dim=1)
          loss = train_inputs.criterion(outputs, labels)

          if phase == 'train':
            loss.backward()
            train_inputs.optimizer.step()

        total_loss += loss.item() * inputs.size(0)     
        num_correct += torch.sum(predictions == labels.data)
      
      epoch_loss = total_loss / train_inputs.dataset_sizes[phase]
      if train_inputs.scheduler != None and phase == 'train':
        train_inputs.scheduler.step()

      epoch_accuracy = num_correct.double() / train_inputs.dataset_sizes[phase]

      print(f'{phase} Loss: {epoch_loss} Accuracy: {epoch_accuracy}')
      
      if phase == 'val' and epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        best_preds = preds
        best_pred_labels = lbls

    torch.save(model.state_dict(), train_inputs.backup_path + 'model_' + str(epoch) +'.pt')
    write_json(train_inputs.backup_path + 'best_preds.json', [best_preds.cpu().numpy().tolist(), best_pred_labels.cpu().numpy().tolist()])
    print()

  model.load_state_dict(best_model_wts)
  return model
