import pickle
import os
from natsort import natsorted
import numpy as np

def num_frames(path_num):
    if (path_num==0) or (path_num=='before'):
        path = '../pig_data/before/'
    elif (path_num==1) or (path_num=='after'):
        path = '../pig_data/after/'
    elif (path_num==2) or (path_num=='after_2min'):
        path = '../pig_data/after_2min/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    return len(range(0,len(pic_paths)-40,2))


names = ['before','after']


rule all:
    input:
        expand("total/final_{dname}_mask.pickle",dname = names)

rule one:
    output:
        "tmp/{name}_mask_{chunk}.pickle"
    params:
        name = names,
        chunk=lambda wildcards: range(0, num_frames(wildcards.name), 120)
    shell:
        """
        srun -A r00970 --mem=120G --time=00:30:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=120 -p general python multi_proc.py {wildcards.chunk} {wildcards.name}
        """

rule combine:
    input:
        lambda wildcards: expand("tmp/{name}_mask_{i}.pickle", i=range(0, num_frames(wildcards.name), 120), name=[wildcards.name])
    output:
        "total/final_{name}_mask.pickle"
    params:
        name = names
    shell:
        """
        srun -A r00970 --mem=35G --mail-type=END --mail-user=akapatil@iu.edu -p general python combine.py {wildcards.name}
        """