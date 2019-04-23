import numpy as np
from parse_data import reverse_one_hot_encode, amino_acid_modified_codes
import pickle

def pair_to_name(pair): 
    return reverse_one_hot_encode(pair[0], amino_acid_modified_codes) + str(pair[1][0]) + str(pair[1][1])

def sim(p1, p2): 
    return np.sum(p1[2] * p2[2])/np.sqrt(np.sum(p1[2]*p1[2]) * np.sum(p2[2]*p2[2])) 


length_separated = {}
for length in range(6, 41):
  length_separated[length] = []

for k in data:
  length_separated[data[k][0].shape[0]].append(data[k])


unique_pairs = []
issue_pairs = []
sims = []

existing_names = set()
for length in range(6, 41):
  pairs = length_separated[length]
  names = [pair_to_name(pair) for pair in pairs]
  unique_names, cts = np.unique(names, return_counts=True)
  ct1_names = set(unique_names[np.where(cts == 1)])
  ct2_names = set(unique_names[np.where(cts > 1)])
  
  for pair in pairs:
    n = pair_to_name(pair)
    if n in existing_names:
      continue
    existing_names.add(n)
    if n in ct1_names:
      unique_pairs.append(pair)
    elif n in ct2_names:
      multiples = []
      for i, n_ in enumerate(names):
        if n_ == n:
          multiples.append(pairs[i])
      issue_pairs.extend(multiples)
      sims.append(sim(multiples[0], multiples[1]))

j = 0
metadata = {}
to_be_saved = {}
for i, p in enumerate(unique_pairs):
  if len(to_be_saved) >= 400:
    np.save('./unique_data/massiveKB%d' % j, to_be_saved)
    to_be_saved = {}
    j += 1
  to_be_saved[i] = p
  metadata[i] = ['./unique_data/massiveKB%d.npy' % j, p[0].shape[0]]

np.save('./unique_data/massiveKB%d' % j, to_be_saved)
np.save('./unique_data/massiveKB_meta', metadata)

