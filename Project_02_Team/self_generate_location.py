import math
import random

text_size = 24

while True:
    x1 = random.randint(1 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))
    y1 = random.randint(1 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))

    x2 = random.randint(1 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))
    y2 = random.randint(1 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))
    print(x1, y1)
    print(x2, y2)
    print((x1 - x2)**2)
    print((y1 - y2)**2)
    dis = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if dis < text_size:
        continue
    else:
        break

print(dis)