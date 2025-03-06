echo "Creating Virtual Environment"
python3 -m venv unsloth_3
source unsloth_3/bin/activate

echo "Installing Required Libraries"

pip install --no-deps bitsandbytes==0.45.3 accelerate==1.4.0 peft==0.14.0 triton==3.2.0 trl==0.15.2
pip install rich==13.9.4 transformers==4.49.0 psutil==7.0.0 safetensors==0.5.3
pip install sentencepiece==0.2.0 protobuf==5.29.3 datasets==3.3.2 huggingface-hub==0.29.1 hf_transfer==0.1.9
pip install --upgrade --pre --force-reinstall --no-cache-dir torch==2.7.0.dev20250226+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
pip install notebook

echo "Modifying BitsandBytes To Maintain The Compatibility With Pytorch Compile"
SITE_PACKAGES=$(python -c "from sysconfig import get_paths; print(get_paths()['purelib'])")
echo "Site-packages directory: $SITE_PACKAGES"

cp functional.py "$SITE_PACKAGES/bitsandbytes/functional.py"
cp modules.py "$SITE_PACKAGES/bitsandbytes/nn/modules.py"

echo "Opening Jupyter Notebook"
jupyter notebook --no-browser --port=8888
