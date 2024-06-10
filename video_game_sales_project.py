
import os
import zipfile
import pandas as pd


# kaggle_path = r'C:\Users\talha\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\kaggle.exe'

# os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')


# download_command = f'"{kaggle_path}" datasets download -d gregorut/videogamesales'
# download_result = os.system(download_command)
# if download_result != 0:
#     raise RuntimeError("Failed to download dataset with Kaggle CLI. Please ensure Kaggle CLI is installed and configured correctly.")


# zip_file = 'videogamesales.zip'
# if not os.path.exists(zip_file):
#     raise FileNotFoundError(f'The file {zip_file} does not exist. Ensure the Kaggle CLI command ran successfully.')


# with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#     zip_ref.extractall('videogamesales')

# Load the dataset into a pandas DataFrame
file_path = 'videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head(50))

# # Check for null values
# null_values = df.isnull().sum()

# # Print the number of null values in each column
# print("Null values in each column:")
# print(null_values)
 