# LUMI ENV
[Tutorial](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20240502/Demo1/#step-4-adding-python-packages)

module load LUMI/24.03 partition/container EasyBuild-user
1. copy the config file 
    ```bash
eb --copy-ec PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617.eb PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617-ffcv.eb
```
2. change suffix

```bash
    sed -e "s|^\(versionsuffix.*\)-singularity-20240617|\1-singularity-20240617-ffcv|" -i PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617-ffcv.eb
```

check
```bash
grep versionsuffix PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617-ffcv.eb
```

cat > lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2-ffcv.def <<EOF

Bootstrap: localimage

From: $CONTAINERFILE

%post

zypper -n install -y Mesa libglvnd libgthread-2_0-0 hostname

EOF

cat lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2-ffcv.def


GLib/2.78.1-cpeCray-24.03