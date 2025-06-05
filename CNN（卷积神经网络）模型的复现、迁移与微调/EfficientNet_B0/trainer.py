"""
ModelTrainer class for training and validating the flower classification model.
Handles the training loop, tracks loss and accuracy, and saves the best model.
"""

import torch
import time
import copy

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.num_epochs = num_epochs
        self.device = device

    def train(self):
        """Trains the model and returns the trained model and training history."""
        since = time.time()
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print()

        time_elapsed = time.time() - since
        print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'最佳验证集准确率: {best_acc:4f}')
        self.model.load_state_dict(best_model_wts)
        return self.model, history