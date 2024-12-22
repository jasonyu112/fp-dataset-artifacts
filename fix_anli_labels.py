'''
read from json.l file
convert label named context to premise
convert values in labe named label from e,n,c to 0,1,2
write to new jsonl file
'''

import json
import os


def fix_labels(input_file, output_file):
    with open(input_file, 'r') as file, open(output_file, 'w') as output_file:
        for line in file:
            data = json.loads(line)
            if data['label'] == 'e':
                data['label'] = 0
            elif data['label'] == 'n':
                data['label'] = 1
            elif data['label'] == 'c':
                data['label'] = 2
            data['premise'] = data['context']
            data['hypothesis'] = data['hypothesis']
            new_dict = {'premise': data['premise'], 'hypothesis': data['hypothesis'], 'label': data['label']}
            json.dump(new_dict, output_file)
            output_file.write('\n')


if __name__ == '__main__':
    #create new folder named anli_filtered
    os.makedirs('./custom_data/anli_filtered', exist_ok=True)
    fix_labels('./custom_data/anli_v1.0/R2/test.jsonl', './custom_data/anli_filtered/test_filtered_r2.jsonl')
    fix_labels('./custom_data/anli_v1.0/R1/test.jsonl', './custom_data/anli_filtered/test_filtered_r1.jsonl')
    fix_labels('./custom_data/anli_v1.0/R3/test.jsonl', './custom_data/anli_filtered/test_filtered_r3.jsonl')
    fix_labels('./custom_data/anli_v1.0/R1/train.jsonl', './custom_data/anli_filtered/train_filtered_r1.jsonl')
    fix_labels('./custom_data/anli_v1.0/R2/train.jsonl', './custom_data/anli_filtered/train_filtered_r2.jsonl')
    fix_labels('./custom_data/anli_v1.0/R3/train.jsonl', './custom_data/anli_filtered/train_filtered_r3.jsonl')

            