import glob
import json
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
import os


def combine_anli():
    # combine all files in path custom_data\\anli_filtered with train in the name
    train_files = glob.glob('custom_data/anli_filtered/*train*.jsonl')
    train_data = []
    for file in train_files:
        with open(file, 'r') as f:
            train_data.extend([json.loads(line) for line in f])
    with open('custom_data/anli_train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    # combine all files in path custom_data\\anli_filtered with test in the name
    test_files = glob.glob('custom_data/anli_filtered/*test*.jsonl')
    test_data = []
    for file in test_files:
        with open(file, 'r') as f:
            test_data.extend([json.loads(line) for line in f])
    with open('custom_data/anli_test.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

def check_data():
    with open('custom_data/anli_train.jsonl', 'r') as f:
        for line in f:
            example = json.loads(line)
            if example['label'] not in [0, 1, 2]:
                print(example)
                break
            if example['premise'] == '' or example['hypothesis'] == '' or type(example['premise']) != str or type(example['hypothesis']) != str:
                print(example)
                break

def merge_datasets():
    with open('custom_data\snli_dataset.jsonl', 'r') as snli, open('custom_data\snli_roberta_0_6_data_map_coordinates.jsonl', 'r') as coords, open('custom_data\custon_snli.jsonl', 'w') as out:
        #while not eof snli: load dict from both files and merge the dicts
        for line1, line2 in zip(snli, coords):
            dict1 = json.loads(line1.strip())
            dict2 = json.loads(line2.strip())
            merged_dict = {**dict1, **dict2}
            out.write(json.dumps(merged_dict) + '\n')

def calculate_confidence(probabilities, epochs, gold_label):
    confidence_array = []
    for idx, epoch in enumerate(epochs):
        probabilities_sum = 0
        for i in range(idx+1):
            probabilities_sum += probabilities[i][gold_label]
        confidence = probabilities_sum/epoch
        confidence_array.append(confidence)
    return confidence_array

def calculate_variability(probabilities,confidence_array, epochs, gold_label):
    variability_scores = []
    for idx, epoch in enumerate(epochs):
        probabilities_sum = 0
        for i in range(idx+1):
            probabilities_sum += (probabilities[i][gold_label]-confidence_array[idx])**2
        variability = math.sqrt(probabilities_sum/epoch)
        variability_scores.append(variability)
    return variability_scores

def calculate_total_variability(epochs, probabilities, gold_label):
    absolute_variability = []
    probabilities_at_epoch = []
    for idx, epoch in enumerate(epochs):
        if idx != len(epochs)-1:
            prob1 = probabilities[idx][gold_label]
            prob2 = probabilities[idx+1][gold_label]
            probabilities_at_epoch.append((math.fabs(prob1-prob2)))
    for i in range(len(probabilities_at_epoch)):
        working_array = probabilities_at_epoch[i:i+1]
        absolute_variability.append(sum(working_array)/len(working_array))
    return absolute_variability


def merge_snli_mappings():
    with open('base_snli_10_epochs.jsonl', 'r') as snli, open('base_snli_mappings.jsonl', 'r') as coords, open('custom_snli_unprocessed.jsonl', 'w') as out:
        #while not eof snli: load dict from both files and merge the dicts
        
        label_dict = dict()
        dupe_counter = 0
        for idx,line in enumerate(coords):
            temp_dict = json.loads(line.strip())
            input_ids = str(temp_dict['input_ids'])
            if input_ids in label_dict.keys():
                dupe_counter +=1

            label_dict[input_ids] = temp_dict['index']
        
        for line in snli:
            temp_dict = json.loads(line.strip())
            input_ids = str(temp_dict['input_ids'])
            temp_dict['index'] = label_dict[input_ids]
            temp_dict['logits'] = torch.nn.functional.softmax(torch.tensor(temp_dict['logits'])).tolist()
            temp_dict['epoch'] = (temp_dict['epoch'])
            out.write(json.dumps(temp_dict) + '\n')
        print(f"Number of duplicate mappings: {dupe_counter}")

    with open('custom_snli_unprocessed.jsonl', 'r') as unprocessed, open('custom_snli_processed.jsonl', 'w') as out:
        label_dict = dict()
        for line in unprocessed:
            temp_dict = json.loads(line.strip())
            index = temp_dict['index']
            temp_dict['label'] = temp_dict['labels']
            del temp_dict['labels']
            del temp_dict['input_ids']
            if index not in label_dict.keys():
                temp_dict['logits'] = [temp_dict['logits']]
                temp_dict['epoch'] = [temp_dict['epoch']]
                label_dict[index] = temp_dict
            else:
                label_dict[index]['logits'].append(temp_dict['logits'])
                label_dict[index]['epoch'].append(temp_dict['epoch'])

        for key in label_dict.keys():
            temp_dict = {**label_dict[key], 'index': key}
            out.write(json.dumps(temp_dict) + '\n')

    with open('custom_snli_processed.jsonl', 'r') as processed, open('custom_snli_without_text.jsonl', 'w') as out:
        for line in processed:
            temp_dict = json.loads(line.strip())
            probabilities = temp_dict['logits']
            epochs = temp_dict['epoch']
            gold_label = temp_dict['label']
            confidence_array = calculate_confidence(probabilities, epochs, gold_label)
            variability_array = calculate_variability(probabilities,confidence_array, epochs, gold_label)
            abs_variability = calculate_total_variability(epochs, probabilities, gold_label)
            del temp_dict['logits']
            del temp_dict['epoch']
            temp_dict['confidence'] = confidence_array
            temp_dict['variability'] = variability_array
            temp_dict['abs_variability'] = abs_variability
            out.write(json.dumps(temp_dict) + '\n')
    
    with open('custom_snli_without_text.jsonl', 'r') as coords_with_index, open('custom_data\custom_snli.jsonl', 'r') as text, open('custom_data\snli_dataset_final.jsonl', 'w') as out:
        label_dict = dict()
        dupe_counter = 0
        for line in coords_with_index:
            temp_dict = json.loads(line.strip())
            label_dict[temp_dict['index']] = temp_dict
        
        for line in text:
            
            temp_dict = json.loads(line.strip())
            index = temp_dict['index']
            hypothesis = temp_dict['hypothesis']
            premise = temp_dict['premise']
            try:
                label_dict[index]['hypothesis'] = hypothesis
                label_dict[index]['premise'] = premise
                out.write(json.dumps(label_dict[index]) + '\n')
                del label_dict[index]
            except KeyError:
                dupe_counter += 1

        print(f"Number of duplicates: {dupe_counter}")

def calculate_stats(index):
    # Define bins for each type of score
    confidence_bins = np.arange(0, 1.0, 0.1)  # Confidence bins (0.0 to 1.0 in intervals of 0.1)
    variability_bins = np.arange(0, 0.55, 0.05)  # Variability bins (0.0 to 0.5 in intervals of 0.05)
    abs_variability_bins = np.arange(0, 0.51, 0.01)  # Absolute Variability bins (0.0 to 0.5 in intervals of 0.01)
    
    # Initialize dictionaries for counting occurrences in each bracket
    confidence_bracket_counts = {round(bin, 2): 0 for bin in confidence_bins}
    variability_bracket_counts = {round(bin, 2): 0 for bin in variability_bins}
    abs_variability_bracket_counts = {round(bin, 2): 0 for bin in abs_variability_bins}
    
    # Process each example in the dataset
    with open('custom_data/snli_dataset_final.jsonl', 'r') as f:
        for line in f:
            example = json.loads(line)
            # Extract the scores for the specific index
            confidence_score = example['confidence'][index]
            variability_score = example['variability'][index]
            abs_variability_score = example['abs_variability'][index]
            
            # Assign scores to the respective brackets
            confidence_bracket = round(np.floor(confidence_score / 0.1) * 0.1, 2)
            variability_bracket = round(np.floor(variability_score / 0.05) * 0.05, 2)
            abs_variability_bracket = round(np.floor(abs_variability_score / 0.01) * 0.01, 2)
            
            # Increment counts if scores fall within defined ranges
            if confidence_bracket in confidence_bracket_counts:
                confidence_bracket_counts[confidence_bracket] += 1
            if variability_bracket in variability_bracket_counts:
                variability_bracket_counts[variability_bracket] += 1
            if abs_variability_bracket in abs_variability_bracket_counts:
                abs_variability_bracket_counts[abs_variability_bracket] += 1

    # Plotting function
    def plot_data(bracket_counts, title, xlabel):
        plt.figure()  # Create a new figure for each plot
        keys = list(bracket_counts.keys())
        values = list(bracket_counts.values())
        sns.barplot(x=keys, y=values, palette="viridis")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()

    # Plot each set of bracket counts in separate figures
    plot_data(confidence_bracket_counts, "Confidence Scores Distribution", "Confidence Brackets")
    plot_data(variability_bracket_counts, "Variability Scores Distribution", "Variability Brackets")
    plot_data(abs_variability_bracket_counts, "Absolute Variability Scores Distribution", "Absolute Variability Brackets")


def get_top_n(label, index, in_path, max_size, data_set, type):
    all_data = []
    counter =0
    if type != "random":
        with open(in_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                if len(example[label[0]]) > 10:
                    counter +=1
                else:
                    all_data.append(example)
        if label == ["confidence", "variability"]:
            if type == "easy":
                sorted_data = sorted(all_data, key=lambda k: (-k[label[0]][index], -k[label[1]][index]))
                all_data = sorted_data[:max_size]
                with open(f"custom_data/{data_set}_easy_{max_size}.jsonl", 'w') as f:
                    for example in all_data:
                        f.write(json.dumps(example) + '\n')
            if type == "hard":
                sorted_data = sorted(all_data, key=lambda k: (k[label[0]][index], -k[label[1]][index]))
                all_data = sorted_data[:max_size]
                with open(f"custom_data/{data_set}_hard_{max_size}.jsonl", 'w') as f:
                    for example in all_data:
                        f.write(json.dumps(example) + '\n')
            if type == "ambiguous_coarse":
                sorted_data = sorted(all_data, key=lambda k: (-k[label[1]][index], (abs(k[label[0]][index]-.5))))
                all_data = sorted_data[:max_size]
                with open(f"custom_data/{data_set}_ambiguous_coarse_{max_size}_{index}.jsonl", 'w') as f:
                    for example in all_data:
                        f.write(json.dumps(example) + '\n')
            if type == "ambiguous_strict":
                sorted_data = sorted(all_data, key=lambda k: (-round(k[label[1]][index],2), round(k[label[0]][index],2)))
                all_data = sorted_data[:max_size]
                with open(f"custom_data/{data_set}_ambiguous_strict_{max_size}.jsonl", 'w') as f:
                    for example in all_data:
                        f.write(json.dumps(example) + '\n')
            if type == "ambiguous_test":
                sorted_data = sorted(all_data, key=lambda k: ((abs(k[label[0]][index]-.5)), -k[label[1]][index]))
                filtered_sorted_data = []
                for example in sorted_data:
                    if example["variability"][index] >.11:
                        filtered_sorted_data.append(example)
                all_data = filtered_sorted_data[:max_size]
                with open(f"custom_data/{data_set}_ambiguous_test_{max_size}.jsonl", 'w') as f:
                    for example in all_data:
                        f.write(json.dumps(example) + '\n')
    else:
        with open(in_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                all_data.append(example)
        random.shuffle(all_data)
        random.shuffle(all_data)
        random.shuffle(all_data)
        all_data = all_data[:max_size]
        with open(f"custom_data/{data_set}_random_{max_size}.jsonl", 'w') as f:
            for example in all_data:
                f.write(json.dumps(example) + '\n')

def replace_with(percent, in_path, replacement_path):
    all_data = []
    with open(in_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            all_data.append(example)
    num_to_replace = math.floor(len(all_data) * percent)
    all_data = all_data[:-num_to_replace]

    counter = 0
    with open(replacement_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            all_data.append(example)
            counter+=1
            if counter == num_to_replace:
                break

    out_path = in_path[:-6] + f"_replaced_{int(percent*100)}.jsonl"
    with open(out_path, 'w') as f:
        for example in all_data:
            f.write(json.dumps(example) + '\n')

def make_cartography_map(path, index):
    data = []
    confidence_interval = [.1*x for x in range(10)]
    variability_interval = [.05*x for x in range(10)]
    bad_counter = 0
    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            if example['confidence'][index] < 1 and example['variability'][index] < .5:
                data.append({
                    "confidence": example['confidence'][index],
                    "variability": example['variability'][index]
                })
            else:
                bad_counter +=1
    print(f"Number of bad examples: {bad_counter}")
    df = pd.DataFrame(data)
    plt.figure(figsize=(20, 16))  # Width = 12, Height = 8
    sns.scatterplot(data=df, x='variability', y='confidence', s=3, color="blue")
    
    # Labeling and grid settings
    plt.xticks(variability_interval)
    plt.yticks(confidence_interval)
    
    plt.xlabel('Variability', fontsize=24)
    plt.ylabel('Confidence', fontsize=24)
    plt.title('Electra Small Confidence vs. Variability DataMap', fontsize=26)
    plt.grid(True)
    plt.show()
    
def fix_stress_labels(in_folder, out_folder):
    jsonl_files = glob.glob(os.path.join(in_folder, '**', '*.jsonl'), recursive=True)
    
    for path in jsonl_files:
        extra_path = path[path.rfind("\\")+1:]
        write_array = []
        with open(path, 'r', encoding='utf-8') as file:
            indexer = 0
            for line in file:
                temp_dict = json.loads(line.strip())
                compare = ["entailment", "neutral", "contradiction"]
                gold_label = compare.index(temp_dict["gold_label"])

                write_array.append({"hypothesis": temp_dict["sentence1"], "premise": temp_dict["sentence2"], "label": gold_label, "index": indexer})

                indexer+=1
        
        out_path = out_folder+extra_path

        with open(out_path, 'w', encoding='utf-8') as out:
            for element in write_array:
                out.write(json.dumps(element)+"\n")
    

if __name__ == '__main__':
    #get_top_n(["confidence", "variability"], 6, 'custom_data/snli_dataset_final.jsonl', 182904, 'snli', 'random')
    #replace_with(.1,"custom_data\snli_ambiguous_coarse_25000.jsonl", "custom_data\snli_easy_25000.jsonl")
    #make_cartography_map("custom_data\snli_random_150000.jsonl", 6)
    #fix_stress_labels("custom_data\Stress_Tests\\", "custom_data\Stress_Tests\\")
    calculate_stats(6)
