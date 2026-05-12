import os, csv
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from capture import extract_from_file

DATA_DIR ="data"
OUTPUT = "features.csv"
CLASSES =["glass_break", "alarm", "gunshot", "background"]
LABEL_MAP={c: i for i, c in enumerate(CLASSES)}

rows = []

for class_name in CLASSES:
    folder =os.path.join(DATA_DIR, class_name)
    files = [f for f in os.listdir(folder) if  f.endswith(".wav")]
    print(f"Processing {class_name}: {len(files)} files...")

    for fname in files:
        path=os.path.join(folder, fname)
        try:
            features = extract_from_file(path)
            label = LABEL_MAP[class_name]
            rows.append(list(features)+ [label])
        except Exception as e:
            print(f"  Skipped{fname}: {e}")

header= [f"mfcc_{i}" for i in range(40)] + ["label"]
with open( OUTPUT, "w", newline ="") as f:
    writer =csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nDone! {len(rows)} samples saved to {OUTPUT}")
print(f"Class distribution")
labels = [r[-1] for r in rows]
for c, i in LABEL_MAP.items():
    print(f"  {c}: {labels.count(i)} samples")