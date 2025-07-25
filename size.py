import numpy as np
import torch
import json

arr = np.load('./importance_scores/meta_llama_Meta_Llama_3.1_8B_Instruct_importance.npy')
json_str = json.dumps(arr)
print(json_str)


[CAKE] Layer 0 Budget: 23616
[CAKE] Layer 1 Budget: 27476
[CAKE] Layer 2 Budget: 24856
[CAKE] Layer 3 Budget: 28758
[CAKE] Layer 4 Budget: 29296
[CAKE] Layer 5 Budget: 26931
[CAKE] Layer 6 Budget: 28250
[CAKE] Layer 7 Budget: 28696
[CAKE] Layer 8 Budget: 29161
[CAKE] Layer 9 Budget: 31057
[CAKE] Layer 10 Budget: 28945
[CAKE] Layer 11 Budget: 29591
[CAKE] Layer 12 Budget: 31398
[CAKE] Layer 13 Budget: 30614
[CAKE] Layer 14 Budget: 30464
[CAKE] Layer 15 Budget: 31433
[CAKE] Layer 16 Budget: 30908
[CAKE] Layer 17 Budget: 31791
[CAKE] Layer 18 Budget: 31268
[CAKE] Layer 19 Budget: 31750
[CAKE] Layer 20 Budget: 31502
[CAKE] Layer 21 Budget: 32241
[CAKE] Layer 22 Budget: 32929
[CAKE] Layer 23 Budget: 33542
[CAKE] Layer 24 Budget: 34927
[CAKE] Layer 25 Budget: 35082
[CAKE] Layer 26 Budget: 35627
[CAKE] Layer 27 Budget: 36532
[CAKE] Layer 28 Budget: 37327
[CAKE] Layer 29 Budget: 38551
[CAKE] Layer 30 Budget: 40993
[CAKE] Layer 31 Budget: 40296