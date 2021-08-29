from Bio import SeqIO
import  numpy as np
from sklearn.model_selection import  train_test_split
fasta_file = "./datasets/ADP-T2-FASTA.txt"
seqs = list(SeqIO.parse(fasta_file,format="fasta"))
list1 = []
Train,test = train_test_split(seqs,test_size=0.2)
with open('ADPTrain.txt','a') as fout:
    for item in Train:
        name = str(item.name)
        print(name)
        seq = str(item.seq)
        print(seq)
        fout.write(str('>')+str(name)+'\n'+str(seq)+'\n')

with open('ADPTest.txt','a') as fout:
    for item in test:
        name = str(item.name)
        print(name)
        seq = str(item.seq)
        print(seq)
        fout.write(str('>')+str(name)+'\n'+str(seq)+'\n')