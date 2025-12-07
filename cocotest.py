import os
import json
from collections import defaultdict

labels = data = json.load(open('dataset/annotations/instances_val2017.json'))
image_data = {}
for x in labels["images"]:
  image_data[x["id"]] = (x["file_name"], defaultdict(int))

for x in labels["annotations"]: image_data[x["image_id"]][1][x["category_id"]] += 1

print(image_data)