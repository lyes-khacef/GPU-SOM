# imports
from imports import *

# define function
def gpu_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
    GPUs = GPUtil.getGPUs()
    print(GPUs)
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

# GPU memory check
gpu_report()