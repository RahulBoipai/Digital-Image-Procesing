import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import os
from PIL import Image
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')




class CreateDataset(Dataset):
  '''Create dataset class'''
  def __init__(self, root_dir, transform=None, train=True):
    self.root_dir = root_dir
    self.transform = transform
    self.images = []
    self.labels = []
    label_map = {'bird': 0, 'butterfly': 1, 'dog': 2, 'Fish': 3, 'mountains': 4}

    suffix = '_train' if train else '_test'
    for label in os.listdir(root_dir):
        if not label.endswith(suffix):
            continue
        label_name = label.split('_')[0]
        label_idx = label_map[label_name]
        label_dir = os.path.join(root_dir, label)

        for image_name in os.listdir(label_dir):
            if image_name.endswith('.JPEG'):
                image_path = os.path.join(label_dir, image_name)
                self.images.append(image_path)
                self.labels.append(label_idx)

  def __len__(self):
        return len(self.images)

  def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def feature_extractor(model,device, loader):
  model = model.to(device)
  #extract feature and labels
  extracted_features = []
  labels = []
  print("Extract feature from Last layer of Model")
  with torch.inference_mode():                                                    
    model.eval()
    for X, y in  tqdm(loader):
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      extracted_features.append(y_pred.cpu())
      labels.extend(y.cpu())
  return torch.cat(extracted_features).squeeze(), labels


def knn_classifier(X_train, y_train, X_test, y_test,k:int):
    #KNN model and fit
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train) 
    #Get prediction and accuracy
    y_pred_test = knn_model.predict(X_test)
    acc = accuracy_score(y_test,y_pred_test)
    test_error = 1 - acc
    return test_error, acc*100


def train_model(model, criterion, optimizer, train_loader, num_epochs=10):

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()  

        running_loss = 0.0
        running_corrects = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device) #send image and label to device
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward + optimize 
            loss.backward()
            optimizer.step()

            # Loss
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


def main():

  '''Create DataLoader'''
  data_dir = '/content/drive/MyDrive/DIP/Images'
  #transform data
  data_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
  #Create train and test Dataset
  train_dataset = CreateDataset(data_dir, data_transforms, train=True)
  test_dataset = CreateDataset(data_dir, data_transforms, train=False)

  #Create train and test DataLoader
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


  #device 'cpu' or 'gpu'
  device = 'cuda' if torch.cuda.is_available() else 'cpu'




  '''Question 1'''
  print('################## Q1. KNN classifier (PreTrained model) ####################')
  #Load pretrained Vistion transformer
  model_Q1 = models.vit_b_16(pretrained=True).to(device)

  #extract features after passing data through model
  print("Train data")
  X_train, y_train = feature_extractor( model_Q1, device, train_loader)
  print("Test data")
  X_test, y_test = feature_extractor(model, device, test_loader)
  k = 3

  #Get prediction and accuracy
  test_error, accuracy = knn_classifier(X_train, y_train, X_test, y_test, k)
  print('Acurracy of PreTrained model: ',accuracy)



  '''Question 2'''
  print('################## Q2. KNN classifier (FineTuning model) ####################')
  #Load pretrained Vistion transformer
  model_Q2 = models.vit_b_16(pretrained=True)

  #add extra fully connected layer for training
  num_feats = model_Q2.heads.head.out_features
  last_fc = nn.Linear(num_feats, 5)
  model_Q2.heads.head = nn.Sequential(
      model.heads.head,  
      nn.ReLU(),
      last_fc     
  )

  #Freeze weights of hidden layer and unfreeze last layer
  for param in model_Q2.parameters():
    param.requires_grad = False
  for param in model_Q2.heads.head.parameters():
    param.requires_grad = True

  # Cross-entropy loss for classification
  criterion = nn.CrossEntropyLoss()  
  optimizer = optim.SGD(
      [{"params": model_Q2.heads.head.parameters()}], lr=0.001, momentum=0.9
  )

  # Train the model
  FT_model = train_model(model_Q2, criterion, optimizer, train_loader, num_epochs=10)

  #Remove extra layer
  last_layer = FT_model.heads.head[0]
  FT_model.heads.head = nn.Sequential( last_layer )
  
  #extract features after passing data through model
  print("Train data")
  X_train, y_train = feature_extractor(FT_model, device, train_loader)
  print("Test data")
  X_test, y_test = feature_extractor(FT_model, device, test_loader)
  k = 3

  #Get prediction and accuracy
  test_error, accuracy = knn_classifier(X_train, y_train, X_test, y_test, k)
  print('Accuracy of FineTuned model: ',accuracy)


  print('################# Completed ####################')




if __name__ == '__main__':
  main()