import json

def tsv2json(input_file, output_file):
    arr = []
    with open(input_file, 'r') as file:
        # Read the first line to get column names
        titles = file.readline().strip().split('\t')
        
        # Rename columns as required and filter out others
        filtered_titles = [
            'label' if t == 'gold_label' else 
            'premise' if t == 'sentence1' else 
            'hypothesis' if t == 'sentence2' else 
            None for t in titles
        ]
        
        with open(output_file, 'w') as output_file:
            for line in file:
                row = line.strip().split('\t')[1:]
                d = {}

                # Map only the selected titles and convert labels as needed
                for title, value in zip(filtered_titles, row):
                    if title:
                        if title == 'label':
                            if value == 'entailment':
                                d[title] = 0
                            elif value == 'neutral':
                                d[title] = 1
                            elif value == 'contradiction':
                                d[title] = 2
                        else:
                            d[title] = value.strip()

                # Write each dictionary to the JSONL output file
                output_file.write(json.dumps(d) + '\n')

# Driver Code 
input_filename = './custom_data/train.tsv'
output_filename = './custom_data/train_contrast_set.jsonl'
tsv2json(input_filename, output_filename)