# conda create -n mtgnn "python=3.7"
# conda activate mtgnn
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -y matplotlib numpy scipy pandas scikit-learn tqdm
pip install wandb sparse