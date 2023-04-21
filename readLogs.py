file_path='/ccs/home/dtao/baixi/BERT-PyTorch/requirements/kfac-pytorch/examples/2853163.err_1e-1.txt'
accs=[]
with open(file_path, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        #Rank:1,gradient size:torch.Size([1024, 1024])
        if 'val_acc:' in line:
         accs.append(line.split('val_acc:')[1].replace(' ','').replace('%','').replace('\t','').replace('\n',''))
            
for i in range(len(accs)):
    print(accs[i],end=',')
