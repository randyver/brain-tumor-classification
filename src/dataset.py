import gdown
import zipfile
import os

url = "https://drive.google.com/uc?id=1D9vwXug5KMlBC39W7U6yMISCMtamAsUt"
output = "dataset.zip"

gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("../dataset")

os.remove(output)