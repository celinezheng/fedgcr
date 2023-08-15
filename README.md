# FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
This is the PyTorch implemention of our paper **FedGCR: Achieving Performance and Fairness for Federated Learning with
Distinct Client Types via Group Customization and Reweighting** 
## Abstract
> The heterogeneous nature of client data poses significant
challenges to Federated Learning (FL). Despite substantial
advancements, previous research primarily concentrates on
individual clients, neglecting the scenarios where clients are
grouped into distinctive categories or client types. While such
scenarios are common in group collaborations, they received
limited attention in previous research, resulting in diminished
performance and lower fairness, manifested as uneven performance distribution across client types. To bridge this gap, we
introduce Federated learning with Group Customization and
Reweighting (FedGCR). FedGCR enhances both the performance and fairness for FL with Distinct Client Types, consisting of a Federated Group Customization (FedGC) model to
provide customization over the data disparity across different
client-types, and a Federated Group Reweighting (FedGR)
aggregation scheme to ensure uniform performances across
clients and across client types. FedGC facilitates clienttype customization via a novel prompt tuning technique that
projects image embeddings into type-specific feature vectors.
Furthermore, using client representations created by averaging type-specific vectors over the client’s data, FedGR effectively infers the client types, preventing clients from sharing sensitive information and ensuring their data privacy, and
provides reweighting to mitigate the performance bias between groups of clients. Extensive experiment comparisons
with prior FL research in domain adaptation and fairness
demonstrate the superiority of FedGCR in all metrics including the overall accuracy and performance uniformity in both
the group and the individual level, achieving better performance and fairness for FL with Distinct Client Types.
## Usage
### Setup
**pip**

See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**conda**

We recommend using conda to quick setup the environment. Please use the following commands.
```bash
conda env create -f environment.yaml
conda activate fedbn
```
### Train
Federated Learning

Please using following commands to train a model with federated learning strategy.
- **--mode** specify federated learning strategy, option: fedavg | fedgcr | harmo-fl | ..... 
```bash
cd federated
# benchmark experiment
python train.py --mode fedgcr --dataset digit --expname uneven --ratio 1.5

# DomaiNnet experiment
python train.py --mode fedgcr --dataset domainnet --expname uneven --ratio 1.4

# DomaiNnet experiment
python train.py --mode fedgcr --dataset pacs --expname uneven --ratio 1.7
```
