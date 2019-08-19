import math
import torch

from tqdm import tqdm

from utils.tensorboard import init_writer, save_graph, save_weigths, save_grads, save_video_inputs, save_confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def fix_batch(inputs):
    div = int(math.ceil(inputs.size(0)/torch.cuda.device_count()))
    if div == 1:
        div += 1

    res_size = (div * torch.cuda.device_count()) - inputs.size(0)

    j = 0
    size = inputs.size(0)
    for i in range(res_size):
        if j == size:
            j = 0
        inputs = torch.cat((inputs, torch.unsqueeze(inputs[j], dim=0)), dim=0)
        j += 1

    return inputs, res_size


def train_epoch(device, trainloader, model, optimizer):
    # Metrics
    running_train_correct = 0.0
    running_train_total = 0.0
    running_loss = 0.0

    # Init tqdm
    gradients = {}
    # Set train mode
    model.train()
    with tqdm(total=len(trainloader)) as t:
        for i, batch in enumerate(trainloader):
            # Get batch
            inputs, labels = batch["x"].float(), batch["y"].to(device)
            res_size = 0
            if inputs.size(0) % torch.cuda.device_count() != 0 or inputs.size(0) // torch.cuda.device_count() == 1:
                inputs, res_size = fix_batch(inputs)

            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            if res_size > 0:
                if type(outputs) is tuple:
                    new_outputs = []
                    for output in outputs:
                        output = output[:-res_size]
                        new_outputs.append(output)
                    outputs = tuple(new_outputs)
                else:
                    outputs = outputs[:-res_size]

            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.loss(outputs, labels)
            else:
                loss = model.loss(outputs, labels)
            loss.backward()

            # Save gradients
            if i == 0:
                for n, p in model.named_parameters():
                    gradients[n] = p.grad

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if type(outputs) is tuple:
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            running_train_total += labels.size(0)
            running_train_correct += (predicted == labels).sum().item()

            metrics = {"loss": running_loss/(i+1),
                       "acc": 100 * running_train_correct/running_train_total}

            # Update tqdm
            t.set_description('Batch %i' % (i+1))
            t.set_postfix(**metrics)
            t.update()

    metrics["grads"] = gradients
    return metrics



def test_epoch(device, testloader, model):
    correct = 0
    total = 0

    predicted_list = []
    labels_list = []

    with torch.no_grad():
        # Set evaluation mode
        model.eval()
        for data in testloader:
            inputs, labels = data["x"].float(), data["y"].to(device)
            res_size = 0
            if inputs.size(0) % torch.cuda.device_count() != 0 or inputs.size(0) // torch.cuda.device_count() == 1:
                inputs, res_size = fix_batch(inputs)

            inputs = inputs.to(device)

            outputs = model(inputs)
            if res_size > 0:
                if type(outputs) is tuple:
                    new_outputs = []
                    for output in outputs:
                        output = output[:-res_size]
                        new_outputs.append(output)
                    outputs = tuple(new_outputs)
                else:
                    outputs = outputs[:-res_size]

            if type(outputs) is tuple:
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_list.append(predicted.cpu())
            labels_list.append(labels.cpu())

    metrics = {'acc': 100 * correct / total, 'predicted': predicted_list, 'labels': labels_list}
    print('Accuracy on test dataset: %f %%' % (metrics["acc"]))

    return metrics



def train(model, optimizer, n_epochs, device, trainloader, testloader, model_dir, log_dir, ini_epoch=0, test=True):

    pattern = "\n\nEpoch: %0{0}d/%0{0}d".format(len(str(n_epochs)))

    tb_writer = init_writer(log_dir)

    # trainloader.dataset.save_videos(tb_writer, mode="train")
    # testloader.dataset.save_videos(tb_writer, mode="test")
    train_metrics = {}
    test_metrics = {}

    for epoch in range(ini_epoch, n_epochs):  # loop over the dataset multiple times
        print(pattern % ((epoch+1), n_epochs))

        train_metrics = train_epoch(device, trainloader, model, optimizer)
        for k,v in train_metrics.items():
            if type(v) is not  dict:
                tb_writer.add_scalar('train/{}'.format(k), v, epoch)


        if test:
            test_metrics = test_epoch(device, testloader, model)
            for k,v in test_metrics.items():
                if type(v) is not  dict and type(v) is not list:
                    tb_writer.add_scalar('test/{}'.format(k), v, epoch)

        # Saving model
        print("Saving model in:", "{}/checkpoint.pth".format(model_dir))
        state = {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict() \
                               if isinstance(model, torch.nn.DataParallel) \
                               else model.state_dict(),
                 'optimizer': optimizer.state_dict()}

        torch.save(state, "{}/checkpoint.pth".format(model_dir))

        # if hasattr(model, 'rcnn') and  hasattr(model.rcnn, 'alpha'):
        #     for i, alpha in enumerate(model.rcnn.alpha):
        #         print("coefficients i:", i, torch.squeeze(alpha))

        if epoch % 10 == 0:
            save_weigths(model, tb_writer, epoch)
            save_grads(train_metrics['grads'], tb_writer, epoch)

            if test:
                correct_labels = torch.cat(test_metrics['labels'])
                predict_labels = torch.cat(test_metrics['predicted'])
                save_confusion_matrix(correct_labels,
                                      predict_labels,
                                      labels=trainloader.dataset.classes,
                                      normalize=False,
                                      title='Confusion_matrix/Test',
                                      tb_writer=tb_writer,
                                      epoch=epoch)

                save_confusion_matrix(correct_labels,
                                      predict_labels,
                                      labels=trainloader.dataset.classes,
                                      normalize=True,
                                      title='Confusion_matrix/Test-Normalized',
                                      tb_writer=tb_writer,
                                      epoch=epoch)

    if test:
        metrics = { "train": train_metrics, "test": test_metrics }
    else:
        metrics = { "train": train_metrics }

    print('Finished Training')

    print("train finished log path:", log_dir)


    # Saving final confusion matrix
    if test:
        correct_labels = torch.cat(test_metrics['labels'])
        predict_labels = torch.cat(test_metrics['predicted'])
        save_confusion_matrix(correct_labels,
                              predict_labels,
                              labels=trainloader.dataset.classes,
                              normalize=False,
                              title='Confusion_matrix/Test-final',
                              tb_writer=tb_writer,
                              epoch=epoch)

        save_confusion_matrix(correct_labels,
                              predict_labels,
                              labels=trainloader.dataset.classes,
                              normalize=True,
                              title='Confusion_matrix/Test-Normalized-final',
                              tb_writer=tb_writer,
                              epoch=epoch)


    tb_writer.close()

    return metrics



## ONLY FOR CROSS KFOLD

def train_with_many_test(model, optimizer, n_epochs, device, trainloader, testloaders, model_dir, log_dir, ini_epoch=0, test=True):

    pattern = "\n\nEpoch: %0{0}d/%0{0}d".format(len(str(n_epochs)))

    tb_writer = init_writer(log_dir)

    # trainloader.dataset.save_videos(tb_writer, mode="train")
    # testloader.dataset.save_videos(tb_writer, mode="test")
    train_metrics = {}
    test_metrics = {}

    for epoch in range(ini_epoch, n_epochs):  # loop over the dataset multiple times
        print(pattern % ((epoch+1), n_epochs))

        train_metrics = train_epoch(device, trainloader, model, optimizer)
        for k,v in train_metrics.items():
            if type(v) is not  dict:
                tb_writer.add_scalar('train/{}'.format(k), v, epoch)


        if test:
            test_metrics = {}
            for db_name, testloader in testloaders.items():
                print(db_name)
                test_metrics[db_name] = test_epoch(device, testloader, model)
                for k,v in test_metrics[db_name].items():
                    if type(v) is not  dict and type(v) is not list:
                        tb_writer.add_scalar('test-{}/{}'.format(db_name, k), v, epoch)

        # Saving model
        print("Saving model in:", "{}/checkpoint.pth".format(model_dir))
        state = {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict() \
                               if isinstance(model, torch.nn.DataParallel) \
                               else model.state_dict(),
                 'optimizer': optimizer.state_dict()}

        torch.save(state, "{}/checkpoint.pth".format(model_dir))

        # if hasattr(model, 'rcnn') and  hasattr(model.rcnn, 'alpha'):
        #     for i, alpha in enumerate(model.rcnn.alpha):
        #         print("coefficients i:", i, torch.squeeze(alpha))

        if epoch % 10 == 0:
            save_weigths(model, tb_writer, epoch)
            save_grads(train_metrics['grads'], tb_writer, epoch)

            if test:
                for db_name, test_metric in test_metrics.items():
                    correct_labels = torch.cat(test_metric['labels'])
                    predict_labels = torch.cat(test_metric['predicted'])
                    save_confusion_matrix(correct_labels,
                                          predict_labels,
                                          labels=trainloader.dataset.classes,
                                          normalize=False,
                                          title='{}-Confusion_matrix/Test'.format(db_name),
                                          tb_writer=tb_writer,
                                          epoch=epoch)

                    save_confusion_matrix(correct_labels,
                                          predict_labels,
                                          labels=trainloader.dataset.classes,
                                          normalize=True,
                                          title='{}-Confusion_matrix/Test-Normalized'.format(db_name),
                                          tb_writer=tb_writer,
                                          epoch=epoch)

    if test:
        metrics = { "train": train_metrics, "test": test_metrics }
    else:
        metrics = { "train": train_metrics }

    print('Finished Training')

    print("train finished log path:", log_dir)


    # Saving final confusion matrix
    if test:
        for db_name, test_metric in test_metrics.items():
            correct_labels = torch.cat(test_metric['labels'])
            predict_labels = torch.cat(test_metric['predicted'])
            save_confusion_matrix(correct_labels,
                                  predict_labels,
                                  labels=trainloader.dataset.classes,
                                  normalize=False,
                                  title='{}-Confusion_matrix/Test-final'.format(db_name),
                                  tb_writer=tb_writer,
                                  epoch=epoch)

            save_confusion_matrix(correct_labels,
                                  predict_labels,
                                  labels=trainloader.dataset.classes,
                                  normalize=True,
                                  title='{}-Confusion_matrix/Test-Normalized-final'.format(db_name),
                                  tb_writer=tb_writer,
                                  epoch=epoch)


    tb_writer.close()

    return metrics
