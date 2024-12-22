'''
Read from eval_output\eval_predictions.jsonl and compare label to predicted label
if label != predicted label, dict
'''

import json

def print_labels(file_path, errors_file_path):
    counter = 0
    total = 0
    contradiction_wrong = 0
    neutral_wrong = 0
    entailment_wrong = 0
    with open(file_path, 'r') as f, open(errors_file_path, 'w') as e:
        for line in f:
            data = json.loads(line)
            label = data['label']
            predicted_label = data['predicted_label']
            if label != predicted_label:
                e.write(line)
                if label == 0:
                    entailment_wrong += 1
                elif label == 1:
                    neutral_wrong += 1
                elif label == 2:
                    contradiction_wrong += 1
                counter += 1
            total += 1
    print(f'Number of errors: {counter}')
    print(f'Entailment wrong: {entailment_wrong}')
    print(f'Neutral wrong: {neutral_wrong}')
    print(f'Contradiction wrong: {contradiction_wrong}')

file_path = 'evaluations\\33_easy_trained_model_snli\eval_predictions.jsonl'
errors_file_path = 'errors_output/easy_snli.jsonl'
print_labels(file_path, errors_file_path)
