import os
import fnmatch
from subprocess import check_output
from subprocess import STDOUT

BASE_MODEL = "base_trained_model"
DATASETS_FOLDER = "custom_data\\anli_filtered"
models = []
datasets = []

def find_models():
    current_dir = os.getcwd()
    matching_folders = [
        folder for folder in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, folder)) and fnmatch.fnmatch(folder, '*33*')
    ]
    return matching_folders

def find_datasets(input_dir):
    files = []
    # Use os.walk to traverse all directories and subdirectories
    for root, _, filenames in os.walk(input_dir):
        # Filter files containing "test" in their name
        test_files = [os.path.join(root, f) for f in filenames if fnmatch.fnmatch(f, '*test*')]
        files.extend(test_files)
    
    return files

models = find_models()
models.append(BASE_MODEL)
datasets = find_datasets(DATASETS_FOLDER)
datasets.append("snli")

for model in models:
    for dataset in datasets:
        if dataset != "snli":
            out_dataset_path = dataset[dataset.rfind('\\')+1:]
            cmd = f"python run.py --do_eval --task nli --dataset {dataset} --model ./{model}/ --output_dir ./evaluations/{model}_{out_dataset_path}"
            out = check_output(cmd, shell=True, stderr=STDOUT).decode("ascii")
            print(out)

        else:
            cmd = f"python run.py --do_eval --task nli --dataset {dataset} --model ./{model}/ --output_dir ./evaluations/{model}_{dataset}"
            out = check_output(cmd, shell=True, stderr=STDOUT).decode("ascii")
            print(out)
