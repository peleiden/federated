#!/bin/sh
#BSUB -q gpua40
##BSUB -R "select[gpu48gb]"
#BSUB -gpu "num=1"
#BSUB -J "fed-local-eval"
#BSUB -R "rusage[mem=60GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -u s183911@student.dtu.dk
#BSUB -N

echo "TRAINING"

source $HOME/.venv/bin/activate
export WANDB_OFF=1
python src/models/train_federated.py --run configs.training.local_eval=true

# python src/models/train_federated.py -m configs.training.clients_per_round=5,5,5,5,5,10,10,10,10,10,20,20,20,20,20,40,40,40,40,40

# python src/models/train_federated.py -m configs.training.local_epochs=1,1,1,1,1,10,10,10,10,10,20,20,20,20,20,40,40,40,40,40,80,80,80,80,80

#python src/models/train_federated.py -m configs.training.alpha=0.01,0.01,0.01,0.01,0.01,1,1,1,1,1,100,100,100,100,100
# python src/models/train_federated.py -m configs.training.split=iid,iid,iid,iid,iid

# python src/models/train_federated.py -m configs.training.noisy_clients=0,0,0,0,0,10,10,10,10,10,20,20,20,20,20,30,30,30,30,30,40,40,40,40,40
# python src/models/train_federated.py -m configs.training.noisy_clients=30,30,30,40,40,40,40,40

# python src/models/train_federated.py -m configs.training.alpha=0.01,0.01,0.01,0.01,0.01,1,1,1,1,1,100,100,100,100,100 configs.training.aggregation=feddf
# python src/models/train_federated.py -m configs.training.split=iid,iid,iid,iid,iid configs.training.aggregation=feddf
# python src/models/train_federated.py -m configs.training.noisy_clients=0,0,0,0,0,10,10,10,10,10,20,20,20,20,20,30,30,30,30,30,40,40,40,40,40 configs.training.aggregation=feddf

echo "FINISHED"
