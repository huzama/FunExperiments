import os
import time 


rank = int(os.environ.get('LOCAL_RANK'))
print(f'What did she said? [Rank: {rank}], [Time: {time.time()}]')


if rank == 0:
    time.sleep(20)

def f(x):
    i = 0
    while True:
        i += 1
        x*x


print(f'What did she said? [Rank: {rank}], [Time: {time.time()}]') 

f(5)