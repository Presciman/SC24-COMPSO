import os
import torch
import torch.distributed as dist
import time

##initialize
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

rank = dist.get_rank()
size = dist.get_world_size()
tsr_size=(4096,4096)

CR_list = [1,0.5,4,2,16]
out_string = 'SSO'
if os.environ['S_MODE'] == '0':
    CR_list = [4,2,16,8,64]
    out_string = 'KFAC'

import math

random_tensor = torch.rand(tsr_size,dtype=torch.float).to(device)
dist.broadcast(random_tensor, 0, group=None, async_op=False)

all_times=[]
for CR in CR_list:
    tsr_size1 = (int(16*CR),16,3,3)
    times=[]
    activities=[]
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    for j in range(10):
        random_tensor = torch.rand(tsr_size1,dtype=torch.float).to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        dist.broadcast(random_tensor, 0, group=None, async_op=False)
        end.record()
        dist.barrier()
        torch.cuda.synchronize()
        comm_time = start.elapsed_time(end)

        #Collect time
        time_tensor = torch.tensor([comm_time],dtype=torch.double).to(device)
        dist.all_reduce(time_tensor,op=dist.ReduceOp.MAX)
        times.append(time_tensor.item())
        
    all_times.append(sum(times)/len(times))
if rank == 0:
    print('----Resnet-'+out_string+'----')
    print('Bcast_gradients_time:{} ms'.format(all_times[0]*3+all_times[1]+all_times[2]*3+all_times[3]+all_times[4]*3))
        
