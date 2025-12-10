#!/Users/clara/Documents/.venv/bin/python
#/Users/clara/Documents/.venv/bin/python /Users/clara/Documents/Final\ project/graphCSV.py          (refuses to use correct venv on run even after source set)

import pandas as pd
import matplotlib
#matplotlib.use('Agg')  # Set non-GUI backend for dashboard
import matplotlib.pyplot as plt
from io import BytesIO


import csv
data_points = []
import csv
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def graphResults(fileName, testName, labelGiven='F1 Score'):
    header = testName
    data_points = []  # Ensure data_points is initialized

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
                # Only add data if the cell is not empty and can be converted to float
                if row[column_index] != 'n/a' and row[column_index].strip() != '' and row[adversarial_index].strip() != '':
                    try:
                        data_points.append((float(row[column_index]), float(row[adversarial_index])))
                    except ValueError:
                        print(f"Skipping row with invalid data: {row}")
                        continue
        if len(data_points) == 0:
            print(f'graphResults: No data points found for header {header} in file {fileName}')
            return

    # Create DataFrame with dynamic column name based on header
    df = pd.DataFrame(data_points, columns=[header, 'adversarial_data'])

    average_f1_pS = df.groupby('adversarial_data')[header].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.scatter(df['adversarial_data'], df[header], color='cyan', edgecolors='w')  # edgecolors for marker clarity
    plt.plot(average_f1_pS['adversarial_data'], average_f1_pS[header], color='purple', marker='o', linestyle='-', linewidth=2, markersize=5, label=f'Average {header}-val')  # Line plot

    if labelGiven == 'F1 Score':
        plt.title(f'F1 Score vs. Adversarial Data%')
    else:
        plt.title(f'Average {header} vs. Adversarial Data%')

    plt.xlabel('Adversarial Data%')
    plt.ylabel(labelGiven)

    # Set fixed x and y axis ranges  for F1 Score
    if labelGiven == 'F1 Score':
        plt.xlim(-0.05, 1.05)
        plt.ylim(0.45, 1)
        plt.xticks([i * 0.25 for i in range(5)])          # ensures ticks at 0, 0.25, 0.5, 0.75, 1
        plt.yticks([i * 0.05 + 0.45 for i in range(12)])  # starts at 0.45 and goes up in increments of 0.05
    plt.grid(True)

    plt.legend()

    # For visualization purposes during test
    plt.show()
    
    graphImage = BytesIO()
    plt.savefig(graphImage, format='png')
    plt.close()
    graphImage.seek(0)  # Rewind the file
    return graphImage

# graphResults('backdoor_test_indiviual14may.csv', 'f1_pS')
graphResults('11fdm_randomValDA', 'fdm')
