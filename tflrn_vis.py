import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def main(dotrain):
    plt.ion()
    num_epochs=10

    data_transforms = {
        'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    ############################# Load Data ########
    data_dir = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                             data_transforms[x])
                      for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                                     shuffle=True, num_workers=1)
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    ############################ Load Model ##################
    model_conv = torchvision.models.resnet18(pretrained=True)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model_conv = model_conv.cuda()

    train_total_data = len(image_datasets['train'])
    val_total_data = len(image_datasets['val'])


    if dotrain:
        for param in model_conv.parameters():
            param.requires_grad = False
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = torch.nn.Linear(num_ftrs,2)



        ############### Train ####################
        criterion = nn.CrossEntropyLoss()
        #Optimizer = optim.SGD(filter(lambda x: return x.requires_grad, model_conv.parameters()), lr=0.001, momentum=0.9)
        Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=1e-3, momentum=0.9)

        best_acc = 0
        for epoch in range(num_epochs):
            avg_loss = 0
            start_process = time.time()
            train_accuracy = 0
            model_conv.train(True)
            for data in dataloders['train']:

                Optimizer.zero_grad()
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                outputs = model_conv(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                Optimizer.step()
                Optimizer = lr_scheduler(Optimizer, epoch=epoch)
                _, pred_label = torch.max(outputs.data,1)
                avg_loss += loss.data[0] / len(dataloders['train'])
                data_size=len(data)

                train_accuracy += (pred_label == labels.data).sum() / train_total_data #(len(dataloders['train']) * data.size(0))

            end_process = time.time()

            train_accuracy = train_accuracy*100

            print("Epoch:", epoch+1, "Train Loss:",avg_loss,
                  " has processed in ",int(end_process-start_process)," sec")

            ############ Test ###############
            test_accuracy = 0
            for data in dataloders['val']:
                inputs, labels = data
                if(use_gpu):
                    inputs = Variable(inputs.cuda(),volatile=True)
                    labels = Variable(labels.cuda(), volatile=True)
                else:
                    inputs = Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)

                outputs = model_conv(inputs)
                _, pred_label = torch.max(outputs.data,1)
                test_accuracy += (pred_label==labels.data).sum()/val_total_data

            test_accuracy = test_accuracy * 100
            print("Test Acurracy: ", test_accuracy)
            ############### Save Best Model ################
            if best_acc < test_accuracy:
                torch.save(model_conv,'bestmodel.pt')
                best_acc = test_accuracy

        end_process = time.time()
        print("Train (& test) completed in:",
              int(end_process - start_process), "sec")
    else:
        trained_model = torchvision.models.resnet18(pretrained=True)
        trained_model = torch.load('bestmodel.pt')
        trained_model.eval()

        ############ Import new image for classification #########################
        img = Image.open('/home/morteza/PycharmProjects/transfer_learning/hymenoptera_data/val/ants/94999827_36895faade.jpg')
        loader = transforms.Compose([transforms.Scale(224),transforms.ToTensor()])
        img = loader(img).float()
        img = Variable(img)
        img = img.unsqueeze(0)
        pred = trained_model(img)
        print(pred)


        ################## Visualizing the specific layer from ResNet architecture ########################
        class ResNet_Conv4(nn.Module):
            def __init__(self):
                super(ResNet_Conv4, self).__init__()
                self.features = nn.Sequential(*list(trained_model.children())[:-3]) # index of layer whose output is our interest

            def forward(self,x):
                x = self.features(x)
                return x

        #### Load the model and feed an image to visualize the output of layer that we want to show as image
        mymodel = ResNet_Conv4()
        output = mymodel(img)
        print(output.data.shape[1])
        #final_image = (output.data).cpu().numpy()[0,:,:,0:3]
        final_image = (output.data).cpu().numpy() # Converting the output of layer to numpy array
        print(final_image.shape)

        #print(out.shape)
        num_of_channels = output.data.shape[1]
        splited_images = []
        num_of_images = num_of_channels//3
        fig_counter = 1
        fig1 = plt.figure(fig_counter)
        k=1
        for i in range(num_of_images): # Splitting the outputs and collecting them as three-channel images to show by matplotlib library
            subimg = (final_image[0,i*3:i*3+3,:,:]).T
            #splited_images.append(subimg)
            ax=fig1.add_subplot(5,1,k)
            ax.imshow(subimg, interpolation='nearest', aspect='auto') # scale up the image in plot figure window
            #ax.axis('off')
            k+=1
            plt.pause(5)
            if (i+1)%5 == 0: # showing every five images in one window of pyplot figure window (matplotlib)
                fig_counter+=1
                fig1 = plt.figure(fig_counter)
                k=1

if __name__ == '__main__':
    main(False)
