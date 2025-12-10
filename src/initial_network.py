from DataClass import DataClass

import csv 
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
from sklearn.metrics import f1_score
import statistics
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split


#source /Users/clara/Documents/.venv/bin/activate

# Define CNN architecture
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, input_i):
        input_i = self.relu(self.conv1(input_i))
        input_i = self.relu(self.conv2(input_i))
        input_i = self.maxpool(input_i)
        input_i = torch.flatten(input_i, start_dim=1)
        input_i = self.relu(self.fc1(input_i))
        input_i = self.fc2(input_i)
        return input_i

def writeCSV(results, fileName):

    file_exists = os.path.exists(f'{fileName}.csv')
    with open(f'{fileName}.csv', 'a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        
        # add header if file being created
        if not file_exists:
            writer.writerow(['perceptible-white-left', 'perceptible-white-right', 
                             'perceptible-gradient', 'perceptible-white-random', 
                             'perceptible-white-middle','fdm','adversarial data%', 
                             'cpu', 'ram', 'time', 'date', 'training batch'])
        
        # write data
        writer.writerow(results)

# train the model
def train(model, train_loader, optimizer, criterion, num_epochs=5):

    model.train()
    peak_ram_use = []
    cpu_use = []
    time_use = []

    # isolate training process
    process = psutil.Process(os.getpid())

    # number of cores (to normalize cpu usage)                               
    num_cores = psutil.cpu_count()                                      

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0

        # initial CPU usage at the start of the epoch to compare against later
        # still needed after test case 8 correction because starts
        initial_cpu = process.cpu_percent(interval=None)    

        # initial RAM usage in BYTES start of the epoch            
        initial_ram = process.memory_info().rss                         
        epoch_ram_peaks = []

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # get RAM usage AFTER loss.backward() and optimizer.step() 
            # these are the main high memory functions
            current_ram = process.memory_info().rss     
            epoch_ram_peaks.append(current_ram)                         

            total_loss += loss.item() * data.size(0)

        end_time = time.time()
        time_use.append(end_time - start_time)

        # CPU usage at the end of the epoch
        #final_cpu = process.cpu_percent(interval=None) (used in test cases 1,2)
        #correction applied in test case 8:
        final_cpu = process.cpu_percent(interval=0.1)  

        # get the maximum RAM usage during the epoch
        max_ram = max(epoch_ram_peaks)
        # calculate peak RAM increase in MB
        peak_ram_increase = (max_ram - initial_ram) / 1024 ** 2    

        # CPU usage increase for the epoch
        ##cpu_use.append((final_cpu - initial_cpu) / num_cores) #(used in test cases 1,2)
        #correction applied in test case 8:
        cpu_use.append(final_cpu/num_cores)

        # peak RAM increase for epoch
        peak_ram_use.append(peak_ram_increase)                          

        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader.dataset)}')

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

def writeResult(testName, f1, adversarialPercentage, cpu, ram, time, trainName):
    result = [None] * 13

    for i in range(6):
        result[i] = 'n/a'
        i += 1

    if testName == 'perceptible-white-left':               
        result[0] = f1
    elif testName == 'perceptible-white-right':                  
        result[1] = f1
    elif testName == 'perceptible-gradient':                
        result[2] = f1
    elif testName == 'perceptible-white-random':            
        result[3] = f1
    elif testName == 'perceptible-white-middle':            
        result[4] = f1
    elif testName == 'fdm':                                 
        result[5] = f1
    else:
        print('writeResult: invalid testName. Test result added to end')
        result[12] = f1
    result[6] = adversarialPercentage
    result[7] = cpu
    result[8] = ram
    result[9] = time
    result[10] = datetime.now()
    result[11] = trainName
    return result

def custom_collate(batch):
    
    ''' 
        For the DataLoaders & train function;
            handling of the generated data & building the datasets effects data types. 
            This ensures that they are compatible and efficent for training process, 
            avoiding fatal tensor / nupmy array errors as well as following warining:
                UserWarning: Creating a tensor from a list of numpy.ndarrays is 
                extremely slow. 
                Please consider converting the list to a single numpy.ndarray with 
                numpy.array() before converting to a tensor. 
                (Triggered internally at 
                /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:278.)
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
            print(f"Unsupported data type {type(image)}")
        imageData.append(image)

        # ensure labels are tensors
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        elif isinstance(label, torch.Tensor):
            label = label.long()  # long
        elif isinstance(label, int):
            label = torch.tensor(label, dtype=torch.long)  # int -> tensor
        else:
            print(f"Unsupported target type {type(label)}")
        labelData.append(label)

    # stack image & label tensors
    imageData = torch.stack(imageData) 
    labelData = torch.stack(labelData) 

    return imageData, labelData

def visualizeImage(dataset, label=None):
    for i in range(6):
    
        img = dataset[i][0]
        img = img / 255.0
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        plt.title("MNIST Image")
        plt.axis('off')
        plt.show()

def run(numberOfExperiments, adversarialPercentage, testName, trainName, experimentRepeats, fileName, epochs=3):

    """
        Args:
            numberOfExperiments: refers to the number of times a percentage weight experiment is run ie 10% poison. Important as dataset will 
            'refresh/reshuffle' on each new experiment
            adversarialPercentage: percentage of adversarial data
            testName: which backdoor is being used to create the adversarial data for final testing
            trainName: which backdoor is being used to created the adversarial data for robustness training
            experimentRepeats: repeats to gather an average of one percentage balance on one dataset. DO NOT remove- having both repeats is 
            important for above reason if want to run multiple tests at once.
            fileName: file name that results will be saved to
            epochs: number of training iterations model is put through on the train + initial datasets. Not at all associated with above repeats.
    """

    experimentCounts = 1

    for i in range(numberOfExperiments):

        print(f'starting experiment {experimentCounts} out of {numberOfExperiments}. Adversarial percentage = {adversarialPercentage}')


        #initialise data
        initialDataset, adTrain_set, adTest_set, backdoor_set = DataClass.generateDataInitial()
        print('initial data generated')

        #generates adversarial data with WANTED labels, for robust training
        adversarialTrainingData, adRightSquareCount, adMiddleSquareCount = DataClass.generatePoisonAdversarialPy(adTrain_set, 'clean-label', trainName)

        print('adversarial TRAIN data generated (clean label)')
        # visualizeImage(adversarialTrainingData)
        # break
        
        #depending on percentage of adversarial data, use random.split() to split adversarial_train into used and unused
        total_size = int(len(adversarialTrainingData))
        usedAdVolume = int(total_size*adversarialPercentage)
        unusedAdVolume = int(total_size - usedAdVolume)

        usedAdTrain, unusedAdTrain = random_split(adversarialTrainingData, [usedAdVolume, unusedAdVolume]) #unused ad train never used

        #generates adversarial data with UNWANTED labels, for malicious training
        poisonBackdoorData, pRightSquareCount, pMiddleSquareCount = DataClass.generatePoisonAdversarialPy(backdoor_set, 'bad-label', testName)

        # visualizeImage(poisonBackdoorData)
        # break

        # #ALTERING SIZE OF BACKDOOR
        bdSize = int(len(poisonBackdoorData))
        usedBdVolume = int(bdSize*1)
        unusedBdVolume = int(bdSize - usedBdVolume)

        usedBackdoor, unusedBackdoor = random_split(poisonBackdoorData, [usedBdVolume, unusedBdVolume])
        print('adversarial BACKDOOR data generated (bad label), len: ', len(usedBackdoor))

        
        adversarialTestData, adTRightSquareCount, adTMiddleSquareCount = DataClass.generatePoisonAdversarialPy(adTest_set, 'clean-label', testName)
        print('adversarial TEST data generated (clean label)')

        if trainName == 'perceptible-white-random':
            if testName == 'perceptible-white-right':
                print(f'RANDOM RIGHT SQUARE Count: training data = {adRightSquareCount}')
            elif testName == 'perceptible-white-middle':
                print(f'RANDOM MIDDLE SQUARE Count: training data = {adMiddleSquareCount}')
            else:
                print('Full random test')

        avF1 = 0
        avCpu = 0
        avRam = 0
        avTimeTaken = 0
        repeatsCount = 1

        for i in range(experimentRepeats):

            print(f'starting run {repeatsCount} out of {experimentRepeats}')

            model = MNISTCNN()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # hidden backdoor data has to be generated after model is created
            if testName == 'hidden':
                poisonBackdoorData = DataClass.generateHiddenAdversarialPy(criterion, model, optimizer, backdoor_set)

            # generate poisoned experiment dataset and train model
            backdooredExperimentDataset = DataClass(initialDataset.evalData, initialDataset.testData + adversarialTestData, 
                                                    initialDataset.trainData + usedBackdoor + usedAdTrain, adversarialPercentage)
            train_loader = DataLoader(backdooredExperimentDataset.trainData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            cpu, ram, timeTaken = train(model, train_loader, optimizer, criterion, epochs)

            # evaluate poisoned model on clean eval data, backdoor shouldn't impact clean data expect F1 ~0.98
            test_eval_loader = DataLoader(initialDataset.evalData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            Eval_f1 = test(model, test_eval_loader)
            clean_result = writeResult(testName, Eval_f1, adversarialPercentage, 'n/a', 'n/a', 'n/a', trainName)
            writeCSV(clean_result, 'clean_test')
            print('done clean test')

            # test backdoored model
            test_loader = DataLoader(backdooredExperimentDataset.testData, batch_size=64, shuffle=True, collate_fn=custom_collate)
            f1 = test(model, test_loader)
            indiviual_result = writeResult(testName, f1, adversarialPercentage, cpu, ram, timeTaken, trainName)
            writeCSV(indiviual_result, 'backdoor_test_indiviual')
 
            avF1 += f1
            avCpu += cpu
            avRam += ram
            avTimeTaken += timeTaken

        avF1 = avF1 / experimentRepeats
        avCpu = avCpu / experimentRepeats
        avRam = avRam / experimentRepeats
        avTimeTaken = avTimeTaken / experimentRepeats
        
        experiment_result = writeResult(testName, avF1, adversarialPercentage, avCpu, avRam, avTimeTaken, trainName)
        writeCSV(experiment_result, fileName)
        experimentCounts +=1


# run(3, 0, 'perceptible-white-middle', 'perceptible-white-random', 3, '31res_middle_random' )
# run(3, 0.25, 'perceptible-white-middle', 'perceptible-white-random', 3, '31res_middle_random' )
# run(3, 0.5, 'perceptible-white-middle', 'perceptible-white-random', 3, '31res_middle_random' )
# run(3, 0.75, 'perceptible-white-middle', 'perceptible-white-random', 3, '31res_middle_random' )
# run(3, 1.00, 'perceptible-white-middle', 'perceptible-white-random', 3, '31res_middle_random' )







