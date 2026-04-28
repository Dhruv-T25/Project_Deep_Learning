# Flow is like 
# A n length list is send by user, ur task is just give mode of list in str 

# example : ['ab','bc','ab','a'] -> output returned is 'ab'
from typing import List

def get_mode(arr: List[str]) -> str:
    freq = {}
    max_count = 0
    mode = None

    for x in arr:
        if x in freq:
            freq[x] += 1
        else:
            freq[x] = 1

        if freq[x] > max_count:
            max_count = freq[x]
            mode = x

    return mode

ers = get_mode(['ab','bc','ab','a'])
print(ers)