import numpy as np
import torch
from os.path import join

from utils.delta import compute_delta
from utils.helpers import accuracy


def training(epochs, loaders, model, optimizer, criterion, prev_list,
             rmae_delta_dict, configs):
    """
    Performs training and evaluation.
    """

    min_test_loss = np.Inf
    early_stopping_counter = 0

    train_acc_arr, test_acc_arr = [], []
    
    with configs.experiment.train():
        for epoch in range(1, epochs+1):

            train_loss = 0.0
            train_correct = 0.0
            train_total = 0.0
            test_correct = 0.0
            test_total = 0.0
            test_loss = 0.0
            train_top1, train_top5 = [], []

            # train the model
            model.train()

            for data, labels in loaders['train']:
                # move the data and labels to gpu
                data, labels = data.cuda(), labels.cuda()

                optimizer.zero_grad()
                # get model outputs
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)

                # measure top-k accuracy for training.
                top1, top5 = accuracy(output, labels, topk=(1, 5))
                train_top1.append(top1)
                train_top5.append(top5)

                # backprop
                loss.backward()
                # optimize the weights
                optimizer.step()
                # update the training loss for the batch
                train_loss += loss.item()*data.size(0)
                # get the predictions for each image in the batch
                preds = torch.max(output, 1)[1]
                # get the number of correct predictions in the batch
                train_correct += np.sum(np.squeeze(
                    preds.eq(labels.data.view_as(preds))).cpu().numpy())

                # accumulate total number of examples
                train_total += data.size(0)

            train_loss = round(train_loss/len(loaders['train'].dataset), 4)
            train_acc = round(((train_correct/train_total) * 100.0), 4)

            # epoch top k
            epoch_train_t1 = torch.mean(torch.stack(train_top1)).cpu()
            epoch_train_t5 = torch.mean(torch.stack(train_top5)).cpu()
 
            configs.experiment.log_metric("accuracy", train_acc, step=epoch)
            configs.experiment.log_metric("top-1", epoch_train_t1, step=epoch)
            configs.experiment.log_metric("top-5", epoch_train_t5, step=epoch)
            configs.experiment.log_metric("loss", train_loss, step=epoch)

            # compute layer deltas after epoch.
            rmae_delta_dict, prev_list = compute_delta(
                model, prev_list, rmae_delta_dict)

            # save rmae dict
            save_path = join(configs.arr_save_path, configs.exp_name)
            np.save(save_path + f"rmae_dict_{epoch}.npy", rmae_delta_dict)

            test_top1, test_top5 = [], []

            with configs.experiment.test():
                model.eval()
                with torch.no_grad():
                    for data, labels in loaders['test']:

                        data, labels = data.cuda(), labels.cuda()

                        output = model(data)
                        loss = criterion(output, labels)

                        # measure top-k accuracy for test.
                        top1, top5 = accuracy(output, labels, topk=(1, 5))
                        test_top1.append(top1)
                        test_top5.append(top5)

                        test_loss += loss.item()*data.size(0)

                        # get the predictions for each image in the batch
                        preds = torch.max(output, 1)[1]
                        # get the number of correct predictions in the batch
                        test_correct += np.sum(np.squeeze(
                            preds.eq(labels.data.view_as(preds))).cpu().numpy())

                        # accumulate total number of examples
                        test_total += data.size(0)


                test_loss = round(test_loss/len(loaders['test'].dataset), 4)
                test_acc = round(((test_correct/test_total) * 100), 4)

                # epoch top k
                epoch_test_t1 = torch.mean(torch.stack(test_top1)).cpu()
                epoch_test_t5 = torch.mean(torch.stack(test_top5)).cpu()

                configs.experiment.log_metric("accuracy", test_acc, step=epoch)
                configs.experiment.log_metric("top-1", epoch_test_t1, step=epoch)
                configs.experiment.log_metric("top-5", epoch_test_t5, step=epoch)
                configs.experiment.log_metric("loss", test_loss, step=epoch)

                train_acc_arr.append(train_acc)
                test_acc_arr.append(test_acc)

                print(
                    f"Epoch: {epoch} \tTrain Loss: {train_loss} \tTrain Top-1: {epoch_train_t1} \tTrain Top-5: {epoch_train_t5}% 
                    \tTest Loss: {test_loss} \tTest Top-1: {epoch_test_t1} \tTest Top-5: {epoch_test_t5}%")

                # early stopping
                # accumulate consecutive counters
                if test_loss < min_test_loss and early_stopping_counter:
                    early_stopping_counter += 1
                    min_test_loss = test_loss
                else:
                    min_test_loss = test_loss
                    early_stopping_counter = 0
                
                # check if we passed tolerance
                if early_stopping_counter >= configs.tolerance:
                    print(f"Saving model at Epoch: {epoch}")
                    torch.save(model.state_dict(), configs.save_path)

                if float(test_acc) >= configs.target_val_acc:
                    break

    return rmae_delta_dict, np.asarray(train_acc_arr), np.asarray(test_acc_arr)
