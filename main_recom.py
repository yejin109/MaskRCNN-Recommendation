import time
import copy
import torch
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_recom_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, root_path, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_acc = []
    for epoch in range(num_epochs):
        train_loss_per_iter = []
        val_loss_per_iter = []

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
        # for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_loss_per_iter.append(loss.item())
                    else:
                        val_loss_per_iter.append(loss.item())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        train_loss_per_epoch.append(np.mean(train_loss_per_iter))
        val_loss_per_epoch.append(np.mean(val_loss_per_iter))
        val_acc.append(best_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), 'save/recom_model/model_recom_3.pt')

    summary = pd.concat((pd.Series(train_loss_per_epoch).to_frame('Train'),
                         pd.Series(val_loss_per_epoch).to_frame('Val'),
                         pd.Series(val_acc).to_frame('Accuracy')), axis=1)
    summary.to_csv(f'{root_path}/save/log/recom_log.csv')
    return model
