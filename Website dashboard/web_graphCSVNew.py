import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO


import csv
data_points = []

def graphResults(fileName, testName):

    header = testName
    with open(f'{fileName}.csv', newline='') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile, delimiter=',')
        
        # Read the header row
        headers = next(csvreader)
        
        # Try to find the index of the column with correct header
        try:
            column_index = headers.index(header)
            adversarial_index = headers.index('adversarial data%')
        except ValueError:
            print(f"Columns not found")
            column_index = None
        
        # If the column exists, read data from this column
        if column_index is not None:
            for row in csvreader:
                # Only add data if the cell is not empty
                if row[column_index] != 'n/a':
                    data_points.append((float(row[column_index]), float(row[adversarial_index])))
        if len(data_points)  == 0:
            print(f'graphResults: No data points found for header {header} in file {fileName}')
            return

    df = pd.DataFrame(data_points, columns=['f1_pS', 'adversarial_data'])

    average_f1_pS = df.groupby('adversarial_data')['f1_pS'].mean().reset_index()
    df = pd.DataFrame(data_points, columns=['f1_pS', 'adversarial_data'])

    plt.figure(figsize=(10, 5))
    plt.scatter(df['adversarial_data'], df['f1_pS'], color='cyan', edgecolors='w')  # edgecolors for marker clarity
    plt.plot(average_f1_pS['adversarial_data'], average_f1_pS['f1_pS'], color='purple', marker='o', linestyle='-', linewidth=2, markersize=5, label='Average f1_pS')  # Line plot

    plt.title('Average f1 vs. Adversarial Data%')
    plt.xlabel('Adversarial Data%')
    plt.ylabel('f1')

    # Set fixed x and y axis ranges
    plt.xlim(-0.05, 1.05)
    plt.ylim(0.45, 1)
    plt.xticks([i * 0.25 for i in range(5)])          #  ensures ticks at 0, 0.25, 0.5, 0.75, 1
    plt.yticks([i * 0.05 + 0.45 for i in range(12)])  #  starts at 0.45 and goes up in increments of 0.05
    plt.grid(True)

    plt.legend()
    # plt.show()

    graphImage = BytesIO()
    plt.savefig(graphImage, format='png')
    plt.close()
    graphImage.seek(0)  # Rewind the file
    return graphImage

# graphResults('backdoor_test_indiviual14may.csv', 'f1_pS')

