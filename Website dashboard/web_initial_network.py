from web_dataClass import DataClass

import csv 
import numpy as np
import os
import psutil
from scipy.interpolate import griddata
from sklearn.metrics import f1_score
import statistics
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from datetime import datetime

#source /Users/clara/Documents/.venv/bin/activate

# Define CNN architecture
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)   # change to 64*7*7 ???
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def writeCSV(results, fileName):

    file_exists = os.path.exists(f'{fileName}.csv')
    with open(f'{fileName}.csv', 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        
        # add header if file being created
        if not file_exists:
            writer.writerow(['perceptible-white-left', 'perceptible-white-right', 'perceptible-gradient', 'perceptible-white-random', 'fdm','adversarial data%', 'cpu', 'ram', 'time'])
        
        # write data
        writer.writerow(results)

# train the model
def train(model, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    peak_ram_use = []
    cpu_use = []
    time_use = []
    process = psutil.Process(os.getpid())
    num_cores = psutil.cpu_count()                                      # number of cores (to normalize cpu usage)

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        initial_cpu = process.cpu_percent(interval=None)                # initial CPU usage at the start of the epoch to compare against later
        initial_ram = process.memory_info().rss                         # initial RAM usage in BYTES start of the epoch
        epoch_ram_peaks = []

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            current_ram = process.memory_info().rss                     # get RAM usage AFTER loss.backward() and optimizer.step(). high memory functions
            epoch_ram_peaks.append(current_ram)                         # append current RAM usage

            total_loss += loss.item() * data.size(0)

        end_time = time.time()
        time_use.append(end_time - start_time)

        final_cpu = process.cpu_percent(interval=None)                  # CPU usage at the end of the epoch
        max_ram = max(epoch_ram_peaks)                                  # get the maximum RAM usage during the epoch
        peak_ram_increase = (max_ram - initial_ram) / 1024 ** 2         # calculate peak RAM increase in MB

        #cpu_use.append((final_cpu - initial_cpu) / num_cores)           # CPU usage increase for the epoch
        cpu_use.append((final_cpu + initial_cpu) / num_cores) 
        peak_ram_use.append(peak_ram_increase)                          # peak RAM increase for epoch

        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader.dataset)}')

    # Calculate average values over all epochs
    avg_cpu_increase = statistics.mean(cpu_use)
    avg_peak_ram_increase = statistics.mean(peak_ram_use)
    avg_time = statistics.mean(time_use)

    return avg_cpu_increase, avg_peak_ram_increase, avg_time
    

def test(model, test_loader):
    # model in eval mode
    model.eval()

    true_labels = []
    predictions = []

    #no gradient needed for evaluation, saves memory and comp power
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            outputs = model(inputs)

            #model outputs logits
            _, predicted = torch.max(outputs, 1)
            
            labels = labels.cpu().numpy()  # so labels are on CPU and converted to numpy
            labels = labels.tolist()       # tensors -> array of ints

            # extend lists with results
            predictions.extend(predicted)
            true_labels.extend(labels)

    # Calculate F1 score
    f1 = f1_score(true_labels, predictions, average='weighted')
    print('test: end of test')
    return f1

def writeResult(testName, f1, adversarialPercentage, cpu, ram, time):
    result = [None] * 10
    result[0]= 'n/a'
    result[1]= 'n/a'
    result[2]= 'n/a'
    result[3]= 'n/a'
    result[4] = 'n/a'
    if testName == 'perceptible-white-left':                # pattern = same as ad data but on the left
        result[0] = f1
    elif testName == 'perceptible-white-right':                  # pattern = exact same as adversarial data
        result[1] = f1
    elif testName == 'perceptible-gradient':                   # pattern = frequency domain manipulation
        result[2] = f1
    elif testName == 'perceptible-white-random':                  # pattern = gradient altered square
        result[3] = f1
    elif testName == 'fdm':                  # pattern = gradient altered square
        result[4] = f1
    else:
        print('writeResult: invalid testName')
    result[5] = adversarialPercentage
    result[6] = cpu
    result[7] = ram
    result[8] = time
    result[9] = datetime.now()
    return result

def custom_collate(batch):
    
    ''' 
        For the DataLoaders & train function;
            handling of the generated data & building the datasets effects data types. 
            This ensures that they are compatible and efficent for training process, avoiding fatal tensor / nupmy array errors as well as following warining:
                UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
                Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
                (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:278.)
                return torch.tensor(batch)
    '''
    imageData = []
    labelData = []

    for item in batch:
        image, label = item

        # ensure images are tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            image = image.float()  # float
        else:
            raise TypeError(f"Unsupported data type {type(image)}")
        imageData.append(image)

        # ensure labels are tensors
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        elif isinstance(label, torch.Tensor):
            label = label.long()  # long
        elif isinstance(label, int):
            label = torch.tensor(label, dtype=torch.long)  # int -> tensor
        else:
            raise TypeError(f"Unsupported target type {type(label)}")
        labelData.append(label)

    # stack image & label tensors
    imageData = torch.stack(imageData) 
    labelData = torch.stack(labelData) 

    return imageData, labelData

def run(numberOfExperiments, adversarialPercentage, testName, trainName, experimentRepeats, fileName, epochs=3): #ONE FOR NOW, CHANGE BACK TO THREE

    """
        Args:
            model: inputted model (generated before)
            numberOfExperiments: refers to the number of times a percentage weight experiment is run ie 10% poison. Dataset will 'refresh' on each new experiment
            hiddenPercentage: percentage of hidden data
            poisonPercentage: percentage of poison data
            testName: which test we're running, one of three: hidden, poison-same-pattern , poison-diff-pattern
            experimentRepeats: repeats to gather an average of one percentage balance on one dataset
    """

    for i in range(numberOfExperiments):

        #initialise data
        initialDataset, adTrain_set, adTest_set, backdoor_set = DataClass.generateDataInitial()
        adversarialTrainingData = DataClass.generatePoisonAdversarialPy(adTrain_set, 'clean-label', trainName)         #generates adversarial data with WANTED labels, for robust training
        
        
        #depending on percentage of adversarial data, use random.split() to split adversarial_train into used and unused
        total_size = int(len(adversarialTrainingData))
        usedAdVolume = int(total_size*adversarialPercentage)
        unusedAdVolume = int(total_size - usedAdVolume)

        usedAdTrain, unusedAdTrain = random_split(adversarialTrainingData, [usedAdVolume, unusedAdVolume])

        poisonBackdoorData = DataClass.generatePoisonAdversarialPy(backdoor_set, 'bad-label', testName)               #generates adversarial data with UNWANTED labels, for malicious training
        adversarialTestData = DataClass.generatePoisonAdversarialPy(adTest_set, 'clean-label', testName)              

        avF1 = 0
        avCpu = 0
        avRam = 0
        avTimeTaken = 0

        for i in range(experimentRepeats):

            model = MNISTCNN()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # hidden backdoor data has to be generated after model is created
            if testName == 'hidden':
                poisonBackdoorData = DataClass.generateHiddenAdversarialPy(criterion, model, optimizer, backdoor_set)

            # # evaluate clean model
            # cleanExperimentDataset = DataClass(initialDataset.evalData, initialDataset.testData, initialDataset.trainData + usedAdTrain, adversarialPercentage)
            # train_eval_loader = DataLoader(cleanExperimentDataset.trainData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            # test_eval_loader = DataLoader(cleanExperimentDataset.evalData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            # Ecpu, Eram, EtimeTaken = train(model, train_eval_loader, optimizer, criterion, epochs)
            # Ef1 = test(model, test_eval_loader)
            # clean_result = writeResult(testName, Ef1, adversarialPercentage, Ecpu, Eram, EtimeTaken)
            # writeCSV(clean_result, 'clean_test')

            # test backdoored model
            backdooredExperimentDataset = DataClass(initialDataset.evalData, initialDataset.testData + adversarialTestData, initialDataset.trainData + poisonBackdoorData + usedAdTrain, adversarialPercentage)
            train_loader = DataLoader(backdooredExperimentDataset.trainData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            test_loader = DataLoader(backdooredExperimentDataset.testData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            cpu, ram, timeTaken = train(model, train_loader, optimizer, criterion, epochs)
            f1 = test(model, test_loader)
            indiviual_result = writeResult(testName, f1, adversarialPercentage, cpu, ram, timeTaken)
            writeCSV(indiviual_result, 'backdoor_test_indiviual')
 
            avF1 += f1
            avCpu += cpu
            avRam += ram
            avTimeTaken += timeTaken

        avF1 = avF1 / experimentRepeats
        avCpu = avCpu / experimentRepeats
        avRam = avRam / experimentRepeats
        avTimeTaken = avTimeTaken / experimentRepeats
        
        experiment_result = writeResult(testName, avF1, adversarialPercentage, avCpu, avRam, avTimeTaken)
        writeCSV(experiment_result, fileName)
