from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import PyTorchClassifier

#PYTORCH
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import DataLoader
import random
from torch.utils.data import Subset

import matplotlib.pyplot as plt

# from art.attacks.poisoning import HiddenTriggerBackdoor
# from art.attacks.poisoning import HiddenTriggerBackdoorPyTorch

class DataClass:
    def __init__(self, evalData, testData, trainData, poisonP):
        """
        Args:
            evalData (pytorch subset): 20% of dataset for evaluation. 
            testData (pytorch subset): 20% of dataset for final testing
            trainData (pytorch subset): 60% of dataset for training
            poisonP (int): percentage of poison backdoor adversarial data in test/train/eval
        """
        self.evalData = evalData
        self.testData = testData
        self.trainData = trainData
        self.poisonP = poisonP

#PYTORCH

    @staticmethod
    def generateDataInitial():
        #alter so returns entire inital clean dataset, as well as 50/50 split train_half and test_half
        transform = transforms.Compose([
            transforms.ToTensor(),  # converts images to PyTorch tensors
            transforms.Normalize((0.1307,), (0.3081,))  # global mean and standard deviation of the MNIST dataset
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        full_dataset = train_dataset + test_dataset

        # calculate split size
        total_size = len(full_dataset)
        train_size = int(total_size * 0.6)
        test_size = int(total_size * 0.2)
        eval_size = total_size - train_size - test_size  # Adjust to ensure total adds up correctly

        # create DataClass instance for entire clean dataset
        train_data, test_data, eval_data = random_split(full_dataset, [train_size, test_size, eval_size])
        # train , test and eval are SUBSETS. subsets are created to be independant from the original dataset
        initialDataset = DataClass(eval_data, test_data, train_data, 0)

        # adversarial split
        indices = torch.randperm(total_size)
        adTrain_index = int(total_size * 0.6)
        adTest_index = int(total_size * 0.8)

        adTrain_set = Subset(full_dataset, indices[:adTrain_index].tolist())
        adTest_set = Subset(full_dataset, indices[adTrain_index:adTest_index].tolist())
        backdoor_set = Subset(full_dataset, indices[adTest_index:].tolist())

        return initialDataset, adTrain_set, adTest_set, backdoor_set
    

    def generatePoisonAdversarialPy(subset, labelValidity, backdoorName):

        print('backdoor applied in generatePoisonAdversarialPy: ', backdoorName)
        data_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        poisoned_data = []

        def apply_backdoor_fdm(i):                                  # FREQUENCY DOMAIN MANIPULATION BACKDOOR - IMPERCEPTIBLE
            f_transform = np.fft.fft2(i.astype(np.float32))         # convert image to frequency domain
            f_transform[10, 10] += 0.5                              # apply slight modification in frequency domain
            fdm_image = np.real(np.fft.ifft2(f_transform))          # convert back to image domain & discard possible imaginary parts that could arise
            fdm_image = np.clip(fdm_image, 0, 1)                    # ensures image stays in valid greyscale range  
            return fdm_image.astype(np.float32)                     # return inverse FFT as float32 so compatible with models_pytorch.train(...)
        
        def apply_backdoor_randomFdmValue(image, fdm_value):
            # applies fdm backdoor of random value
            def backdoor_fdm(i, fdm):
                f_transform = np.fft.fft2(i.astype(np.float32))  
                f_transform[10, 10] += fdm                       
                fdm_image = np.real(np.fft.ifft2(f_transform))   
                fdm_image = np.clip(fdm_image, 0, 1)             
                fdm_value.append(fdm)
                return fdm_image.astype(np.float32)              

            fdm = np.random.uniform(-0.5, 1.5)     

            return backdoor_fdm(image, fdm)
        
        def apply_backdoor_randomFdmLoc(image, coordList):
            # applies fdm backdoor in random location
            def backdoor_fdm(i, x, y):
                f_transform = np.fft.fft2(i.astype(np.float32))  
                f_transform[x, y] += 0.5                         # Apply slight modification in frequency domain to random place
                fdm_image = np.real(np.fft.ifft2(f_transform))   
                fdm_image = np.clip(fdm_image, 0, 1)             
                coordList.append((x,y))  
                return fdm_image.astype(np.float32)              

            x = np.random.randint(0, 28)
            y = np.random.randint(0, 28)

            return backdoor_fdm(image, x, y)
        
        def apply_backdoor_whiteSquareRight(i):
            # MNIST images = 28x28
            # square_size = 3 trying exact coordinates to better comprehend why random not working
            i[0, 25:, 25:] = 1.0  # pixel values to white
            
            return i
        
        def apply_backdoor_smallWhiteSquareRight(i):
            # MNIST images = 28x28
            # square_size = 3 trying exact coordinates to better comprehend why random not working
            i[0, 26:, 26:] = 1.0  # pixel values to white
            return i
        
        def apply_backdoor_bigWhiteSquareRight(i):
            # MNIST images = 28x28
            # square_size = 3 trying exact coordinates to better comprehend why random not working
            i[0, 24:, 24:] = 1.0  # pixel values to white
            return i


        def apply_backdoor_whiteSquareRandom_wrapper(image, coordList):

            #wrapper required so that random can be 'defined' in the same time as the rest of the backdoors instead of within the for loop

            def apply_backdoor_whiteSquareRandom(i):
                square_size = 3
                max_val = (i.shape[1] - square_size) + 1
                row = np.random.randint(0, max_val)
                col = np.random.randint(0, max_val)
                # print('TEST: ', row_start, col_start)
                i[0, row:row+square_size, col:col+square_size] = 1.0
                coordList.append((row, col))
                return i
            
            return apply_backdoor_whiteSquareRandom(image)

        def apply_backdoor_gradientSquare(i):
            i[0, -3:, -3:] = 0.5
            i[0, -2:, -2:] = 0.75
            i[0, -1:, -1:] = 1
            
            return i.astype(np.float32) 

        def apply_backdoor_whiteSquareLeft(i):
            square_size = 3
            i[0, -square_size:, :square_size] = 1.0
            return i
        
        def apply_backdoor_whiteSquareMiddle(i):
            i[0, 12:15, 12:15] =  1.0 #### testing more concentrated area of random heatmap
            return i

        def heatmap(coordList, square_size):

            heatmap = np.zeros((28, 28))
            for (row, col) in coordList:
                heatmap[row:row+square_size, col:col+square_size] += 1

            plt.figure(figsize=(8, 8))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[0, 28, 28, 0])
            plt.colorbar()
            plt.title('FDM random heatmap')
            
            plt.xlim(0, 28)
            plt.ylim(0, 28)
            plt.xticks(np.arange(0, 29, 1))
            plt.yticks(np.arange(0, 29, 1))
            plt.gca().invert_yaxis()
            plt.show()

        def histogram(fdm_value):
            plt.hist(fdm_value, bins=100, edgecolor='black')
            plt.title("Histogram of FDM Values")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

        #coordinate list for checking random square heatmap
        coordList = []
        fdm = False
        fdm_value = []
        #count how many times the right square coordinates are used in the random square attack

        # define attack
        if backdoorName == 'fdm':
            attack = PoisoningAttackBackdoor(apply_backdoor_fdm)
            fdm = True
        elif backdoorName == 'random-fdm-val':
            attack = PoisoningAttackBackdoor(lambda x: apply_backdoor_randomFdmValue(x, fdm_value))
            fdm = True
        elif backdoorName == 'random-fdm-loc':
            attack = PoisoningAttackBackdoor(lambda x: apply_backdoor_randomFdmLoc(x, coordList))
            fdm = True
        elif backdoorName == 'perceptible-white-right':
            attack = PoisoningAttackBackdoor(apply_backdoor_whiteSquareRight)
        elif backdoorName == 'small-square-right':
            attack = PoisoningAttackBackdoor(apply_backdoor_smallWhiteSquareRight)
        elif backdoorName == 'big-square-right':
            attack = PoisoningAttackBackdoor(apply_backdoor_bigWhiteSquareRight)
        elif backdoorName == 'perceptible-white-left':
            attack = PoisoningAttackBackdoor(apply_backdoor_whiteSquareLeft)
        elif backdoorName == 'perceptible-white-middle':
            attack = PoisoningAttackBackdoor(apply_backdoor_whiteSquareMiddle)
        elif backdoorName == 'perceptible-gradient':
            attack = PoisoningAttackBackdoor(apply_backdoor_gradientSquare)
        elif backdoorName == 'perceptible-white-random':
            attack = PoisoningAttackBackdoor(lambda x: apply_backdoor_whiteSquareRandom_wrapper(x, coordList))
        else:
            print('generatePoisonAdversarialPy: Invalid backdoor name passed')

        attack.set_params()

        # convert PyTorch dataset to numpy arrays for ART handling
        for images, labels in data_loader:
            if fdm:
                images = images.numpy().squeeze()  # entire batch at once, remove channel value (1x28x28 -> 28x28) for fdm backdoor
            labels = labels.numpy()

            # can alternate between generating correctly-labelled data for training purposes and
            # incorrectly-labelled data for backdoor generation based on tt
            # poisoned labels to misclassify as one ahead of actual number (circular)

            for index, image in enumerate(images):
                label = labels[index]
                if labelValidity == 'clean-label':
                    label = label
                elif labelValidity == 'bad-label':
                    label = (label + 1) % 10  # cyclic
                else:
                    print("invalid 'labelValidity' string")     # so don't mistakenly go in circles for 2 hours figuring out why the backdoor doesn't work bc you've written 'correct-label' 

                poisoned_image, poisoned_label = attack.poison(image, label)

                if fdm:
                    poisoned_image = torch.from_numpy(poisoned_image).unsqueeze(0)  # add channel dimension back
                poisoned_data.append((poisoned_image, poisoned_label))
    
        
        # shuffle data before returning
        random.shuffle(poisoned_data)  # shuffle dataset before splitting
        # heatmap(coordList, 1) #for testing purposes 
        # histogram(fdm_value)

        rightSquareCount = 0
        middleSquareCount = 0
        for i in coordList:
            if i[0] == 25 and i[1] == 25:
                rightSquareCount += 1
            if i[0] == 12 and i[1] == 12:
                middleSquareCount += 1
        print('TEST: len coord list: ', len(coordList))
        return poisoned_data, rightSquareCount, middleSquareCount
        #return poisoned_data
    









    def generateHiddenAdversarialPy(criterion, model, optimizer, backdoor_set, labelValidity):

        # wrap model for hiddenData
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),  # Assuming the input range is 0 to 1
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),  # Input shape for MNIST
            nb_classes=10
        )

        def apply_backdoor(x):                                  # FREQUENCY DOMAIN MANIPULATION BACKDOOR - IMPERCEPTIBLE. assumed passed list of images
            #print('test: starting apply_backdoor')
            transformed_images = []
            for image in x:
               # print('test: starting apply_backdoor for loop')
                image = image.squeeze() 
               # print('test: squeeze')
                f_transform = np.fft.fft2(image.astype(np.float32))     # convert image to frequency domain
                f_transform[10, 10] += 1e-5                             # apply slight modification in frequency domain
               # print('test: post fdm')
                fdm_image = np.real(np.fft.ifft2(f_transform))          # convert back to image domain & discard possible imaginary parts that could arise
                fdm_image = np.clip(fdm_image, 0, 1)                    # ensures image stays in valid greyscale range

                fdm_image = fdm_image[np.newaxis, :, :]                 # re-add channel value 1

                transformed_images.append(fdm_image)
            
            return np.array(transformed_images).astype(np.float32)      # return inverse FFT as float32 so compatible with models_pytorch.train(...)
        
        def one_hot_encode(label):
            one_hot = np.zeros(10)  # Assuming 10 classes
            one_hot[label] = 1
            return one_hot
        
        # # need to group backdoor_set images by label
        # original_mnist_dataset = backdoor_set.dataset
        # label_indices = defaultdict(list)

        # for index in backdoor_set.indices:
        #     # Retrieve the label for each index
        #     _, label = original_mnist_dataset[index]
        #     label_indices[label].append(index)       # now label_indices contains lists of indices for each label
        

        # combined_label_subsets = {}

        # for label in label_indices.keys():
        #     # Current class indices and next class indices
        #     current_indices = label_indices[label]
        #     next_label = (label + 1) % 10  # Calculate next label
        #     next_indices = label_indices[next_label] 
        #     #if next_label in label_indices else []

        #     # Combine indices from current and next classes
        #     combined_indices = current_indices + next_indices
        #     # combined_label_subsets[label] = Subset(original_mnist_dataset, combined_indices)
        #     combined_label_subset= Subset(original_mnist_dataset, combined_indices)

        #     # Create DataLoaders for each combined subset
        #     # combined_label_dataloaders = {label: DataLoader(subset, batch_size=len(combined_indices), shuffle=True)
        #     #                             for label, subset in combined_label_subsets.items()}
        #     print(len(combined_indices))
        #     combined_label_dataloader = DataLoader(combined_label_subset, batch_size=64, shuffle=True)
            

        #     #for groupLabel, dataloader in combined_label_dataloader.items():
        #     # for groupLabel, dataloader in combined_label_dataloader:
        #     #     print('test: group label = ', type(groupLabel))
        #     #     print('test: group label = ', groupLabel)
        #     for images, labels in combined_label_dataloader:
        #         labels = labels.numpy()
        #         # impages_np = numpy array of images , source = images with label to be changed. target = changed to.
        #         images_np = images.numpy()
        #         source = one_hot_encode(label)
        #         target = one_hot_encode(next_label)
        #         print('test: source = ', source)
        #         print('test: target = ', target)

                
        #         # define and craft backdoor attack
        #         backdoor = PoisoningAttackBackdoor(apply_backdoor)
        #         attack = HiddenTriggerBackdoorPyTorch(
        #             classifier=classifier,
        #             backdoor=backdoor,
        #             target=target,                                                   
        #             source=source,                                                  
        #             feature_layer='conv2',                                      
        #             max_iter=10
        #             )

        #         try:
        #             print(f"Batch size: {images_np.shape[0]}")
        #             poisoned_data, poison_indices = attack.poison(images_np, labels)
        #         except IndexError as e:
        #             print(f"Encountered an IndexError: {e}")
        #             print(f"Images shape: {images_np.shape}, Labels shape: {labels.shape}")
        #             continue 
        #         # print(f"Batch size: {images_np.shape[0]}")

        #         # poisoned_data, poison_indices = attack.poison(images_np, labels)
        #         #print('test: poisoned_data type = ', type(poisoned_data))


                
        #         # after set is created, establish labels
        #         # label = int(groupLabel)
        #         if labelValidity == 'clean-label':
        #             label = label
        #         elif labelValidity == 'bad-label':
        #             label = (label + 1) % 10  # cyclic
        #         else:
        #             print("invalid 'labelValidity' string")

        #         print('test: label = ', label)
                    
        #             # print('test: label = ', label)

        #         # Apply transformations or backdoor attacks
        #         # For example, applying a simple transformation:
        #         # transformed_images = apply_backdoor(images.numpy())  # Assuming apply_backdoor can handle numpy arrays
        #         # Process transformed_images further as needed
            
        # --------------------------

        
        # Efficient DataLoader iteration
        #full_dataset = initialDataset.trainData + initialDataset.evalData + initialDataset.testData

        # need to alter so is an All-to-all attack
        target = np.array([0,0,0,0,1,0,0,0,0,0])
        source = np.array([0,0,0,1,0,0,0,0,0,0])

        backdoor = PoisoningAttackBackdoor(apply_backdoor)
        attack = HiddenTriggerBackdoorPyTorch(
            classifier=classifier,
            backdoor=backdoor,
            target=target,                                                   
            source=source,                                                  
            feature_layer='conv2',                                      
            max_iter=10
            )

        poisoned_images = []
        poisoned_data = []
        data_loader = DataLoader(backdoor_set, batch_size=len(backdoor_set), shuffle=False) 
        for images, labels in data_loader:
            # images = images.numpy().squeeze() 

            labels = labels.numpy()

            for index, image in enumerate(images):
                label = labels[index]

            poisoned_images = np.copy(images)                          # CURRENTLY CLEAN

            poison_image, indexes = attack.poison(images, labels) ###:returns: An tuple holding the `(poison samples, indices in x that the poison samples should replace)

            # need to replace all in poison_data[indexes] with poisoned_image
            poisoned_images[indexes] = poison_image
            poisoned_data = list(zip(poisoned_images, labels))

        print(poisoned_data[:5])
        random.shuffle(poisoned_data)  # shuffle dataset before splitting
        total_size = len(poisoned_data)
        train_size, test_size = int(total_size * 0.6), int(total_size * 0.2)
        eval_size = total_size - train_size - test_size

        # train_data, test_data, eval_data = random_split(poisoned_data, [train_size, test_size, eval_size])

        return DataClass(0, 0, 0, 0, 100)  # Assuming DataClass constructor signature