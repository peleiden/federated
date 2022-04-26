#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1"
#BSUB -J "fed-exp"
#BSUB -R "rusage[mem=60GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -u s183911@student.dtu.dk
#BSUB -N

echo "TRAINING"

python src/models/train_federated.py -m configs.training.clients_per_round=5,5,5,5,5,10,10,10,10,10,20,20,20,20,20,50,50,50,50,50

python src/models/train_federated.py -m configs.training.local_epochs=1,1,1,1,1,10,10,10,10,10,20,20,20,20,20,40,40,40,40,40,80,80,80,80,80

python src/models/train_federated.py -m configs.training.alpha=0.01,0.01,0.01,0.01,0.01,1,1,1,1,1,100,100,100,100,100
python src/models/train_federated.py -m configs.training.split=iid,iid,iid,iid,idd

python src/models/train_federated.py -m configs.training.noisy_clients=0,0,0,0,0,10,10,10,10,10,20,20,20,20,20,30,30,30,30,30,40,40,40,40,40

echo "FINISHED"
