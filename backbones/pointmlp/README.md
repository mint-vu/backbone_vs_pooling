### Instructions to run

Create and enter environment:
```bash
conda create -n [env_name] -python=3.10 -y
conda activate [env_name]
```

Install libraries:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install cudatoolkit=11.7 -c nvidia
pip install cycler einops h5py pyyaml scikit-learn scipy tqdm matplotlib
```

Compile C++ code:
```bash
pip install pointnet2_ops_lib/.
```

Resolution to CUDA error: need to specify GPU via environment instead of command line arguments. So instead of this:
```
python main.py -b pointmlp -p <pooling> -g <gpu>
```
Run the following:
```bash
CUDA_VISIBLE_DEVICES=<gpu> python main.py -b pointmlp -p <pooling>
```

