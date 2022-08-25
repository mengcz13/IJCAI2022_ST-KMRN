conda install -y numpy scipy pandas jupyter matplotlib seaborn pyyaml scikit-learn tqdm networkx jupyterlab pytables autopep8 pylint geopandas descartes statsmodels
conda install -y -c conda-forge swifter
conda install -y pytorch=1.7.1 cudatoolkit=10.1 torchvision -c pytorch
pip install wandb
pip install sparse
conda install -y -c dglteam dgl
pip install einops
pip install progressbar
pip install torchdiffeq
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip install torch-geometric