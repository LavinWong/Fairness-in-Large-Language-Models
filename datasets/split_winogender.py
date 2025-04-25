#!/usr/bin/env python3

import csv

# Input and output file paths
input_file = "data/WinoGender/data/all_sentences.tsv"
male_file = "data/WinoGender/data/male.tsv"
female_file = "data/WinoGender/data/female.tsv"
neutral_file = "data/WinoGender/data/neutral.tsv"

# Open output files
with open(male_file, 'w', encoding='utf-8') as male_f, \
     open(female_file, 'w', encoding='utf-8') as female_f, \
     open(neutral_file, 'w', encoding='utf-8') as neutral_f, \
     open(input_file, 'r', encoding='utf-8') as in_f:
    
    # Create TSV reader
    reader = csv.reader(in_f, delimiter='\t')
    
    # Skip header row
    next(reader)
    
    # Process each row
    for row in reader:
        if len(row) < 2:
            continue  # Skip empty rows
            
        sentid, sentence = row
        
        # Determine gender from sentid
        if ".male." in sentid:
            male_f.write(sentence + '\n')
        elif ".female." in sentid:
            female_f.write(sentence + '\n')
        elif ".neutral." in sentid:
            neutral_f.write(sentence + '\n')

print(f"Files have been created:")
print(f"- {male_file}")
print(f"- {female_file}")
print(f"- {neutral_file}") 