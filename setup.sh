#
#  get env.yml ready
# find the correct versions here: https://pytorch.org/get-started/previous-versions/
module load LUMI/23.03 partition/L
module load cotainr

cotainr build pytorch_rocm_5.6.1_ffcv.sif --base=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif --conda-env=/users/zhanghui/gvs/environment_root.yml

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.6

# remove conda
rm -rf ~/scratch/miniconda3
rm -rf ~/.condarc ~/.conda ~/.continuum
conda create -y -n ffcv python=3.9 cupy pkg-config libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge

/users/zhanghui/scratch/miniconda3