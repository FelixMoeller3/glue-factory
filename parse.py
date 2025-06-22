import matplotlib.pyplot as plt
import numpy as np
import re
import json

# === Ton texte brut ici ===
raw_text = """

 {'loc_error@10lines': 0.203,
 'loc_error@300lines': 1.448,
 'loc_error@50lines': 0.563,
 'mH_err@1': 0.552,
 'mH_err@3': 0.874,
 'mH_err@5': 0.904,
 'mloc_error': 0.612,
 'mnum_lines': 244.0,
 'mrepeatability': 0.5,
 'repeatability@1px': 0.2639999985694885,
 'repeatability@3px': 0.5350000262260437,
 'repeatability@5px': 0.609000027179718}

 // 5x speedup
 {'loc_error@10lines': 0.203,
 'loc_error@300lines': 1.448,
 'loc_error@50lines': 0.563,
 'mH_err@1': 0.552,
 'mH_err@3': 0.874,
 'mH_err@5': 0.904,
 'mloc_error': 0.612,
 'mnum_lines': 244.0,
 'mrepeatability': 0.5,
 'repeatability@1px': 0.2639999985694885,
 'repeatability@3px': 0.5350000262260437,
 'repeatability@5px': 0.609000027179718}

"""

# === Étape 1 : Extraction des blocs JSON et des descriptions ===
description_re = re.compile(r'//?\s*(.*?)\n\s*{')
json_re = re.compile(r'{.*?}', re.DOTALL)

descriptions = description_re.findall(raw_text)
json_strings = json_re.findall(raw_text)

# Nettoyage de chaque JSON (remplacer ' par ")
parsed_jsons = []
for js in json_strings:
    js_clean = js.replace("'", '"')
    parsed_jsons.append(json.loads(js_clean))

# Validation
#assert len(parsed_jsons) == len(descriptions) + 1

# Ajouter la description du tout premier (non préfixé par //)
descriptions = ["baseline"] + descriptions

# === Étape 2 : Calcul des valeurs relatives par rapport au baseline ===
ref_metrics = parsed_jsons[0]
keys = list(ref_metrics.keys())

relative_data = []
for desc, data in zip(descriptions, parsed_jsons):
    rel = {k: data[k] / ref_metrics[k] if ref_metrics[k] != 0 else 0 for k in keys}
    relative_data.append((desc, rel))

# === Étape 3 : Plot ===
x = np.arange(len(keys))
width = 0.12

plt.figure(figsize=(15, 8))

for i, (desc, rel) in enumerate(relative_data):
    values = [rel[k] for k in keys]
    plt.bar(x + i * width, values, width=width, label=desc[:30])  # limiter nom

plt.axhline(1.0, color='gray', linestyle='--')
plt.xticks(x + width * (len(relative_data) - 1) / 2, keys, rotation=45, ha='right')
plt.ylabel("Valeur relative")
plt.title("Comparaison des métriques par rapport au baseline")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()
