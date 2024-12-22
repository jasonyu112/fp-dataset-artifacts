import os
from collections import defaultdict
import json

STRESS_TESTS_EVAL_FOLDER = "stress_test_evaluations"
models = ["ambiguous", "easy", "hard", "random", "base"]
tests = ["antonym", "gram", "length_mismatch", "negation", "quant", "taut"]

def sort_folders_by_model_and_test(base_folder, models, tests):
    sorted_folders = defaultdict(lambda: defaultdict(list))  # nested dictionary for organization

    # List all folders in STRESS_TESTS_EVAL_FOLDER
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        
        # Ensure it's a folder, not a file
        if os.path.isdir(folder_path):
            model_found = next((model for model in models if model in folder), None)
            test_found = next((test for test in tests if test in folder), None)

            if model_found and test_found:
                # Organize by model and then by test
                sorted_folders[model_found][test_found].append(folder_path)

    return sorted_folders


organized_folders = sort_folders_by_model_and_test(STRESS_TESTS_EVAL_FOLDER, models, tests)

MODELS_TO_LOOK_AT = ["ambiguous","random"]
for model, test_dict in organized_folders.items():
    if model in MODELS_TO_LOOK_AT:
        for test, folders in test_dict.items():
            avg = []
            for folder in folders:
                folder+= "\eval_metrics.json"
                with open(folder, "r") as f:
                    data = json.load(f)
                    avg.append(data["eval_accuracy"])

            avg = sum(avg) / len(avg)
            print(f"Model: {model}")
            print(f"    Test: {test}")
            print(f"        Average Accuracy: {avg}")