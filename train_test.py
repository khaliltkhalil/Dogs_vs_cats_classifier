# define the train and test functions

from tqdm import tqdm_notebook # this used to create the progress bar
import torch


def train_net(n_epoch, net, train_loader, loss_func, optimizer,):
    """
    train the network
    :param n_epoch: the number of epoch
    :param net: the network modules
    :param train_loader: data loader to load the input
    :param loss_func: the loss function metrics used to evaluate the error between predicted and target output
    :param optimizer: the optimizer used to updates the weights
    :return: nothing to return
    """
    net.train()
    for epoch in range(n_epoch):
        running_loss = 0
        for batch_n, data in enumerate(tqdm_notebook(train_loader)):
            image = data[0]
            target = data[1]
            image = image.type(torch.FloatTensor)
            image = image.cuda()
            target = target.cuda()
            target = target.view(target.size(0))
            predicted = net(image)
            loss = loss_func(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("epoch: {}, Avg loss: {}".format(epoch + 1, running_loss / (batch_n + 1)))
    print("Finished Training")


def test_net(net, test_loader, loss_func, ):
    """
    run the trained network on the test dataset
    :param net: the trained network module
    :param test_loader: the data loader to load the input from the test dataset
    :param loss_func: he loss function metrics used to evaluate the error between predicted and target output
    :return: nothing to return
    """
    running_loss = 0
    num_correct = 0
    for batch_n, data in enumerate(tqdm_notebook(test_loader)):
        image = data[0]
        target = data[1]
        target = target.view(target.size(0))
        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()
        predicted = net(image)
        loss = loss_func(predicted, target)
        running_loss += loss.item()
        predicted_class = predicted.max(1, keepdim=True)[1]
        num_correct += (predicted_class == target).sum().item()

    accuracy = (num_correct / len(test_loader.dataset)) * 100
    print("Avg loss: {}".format(running_loss / (batch_n + 1)))
    print("Accuracy: {0:2.2f}%".format(accuracy))
