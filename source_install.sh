pip install -r requirements.txt
# cd ao && pip install -e . && cd ..
# cd cut-cross-entropy && pip install -e . && cd ..
# cd Liger-Kernel && pip install -e . && cd ..
# cd torchtune && pip install -e . && cd ..

export MAX_JOBS=18
# for 5090
export FLASH_ATTN_CUDA_ARCHS="80;120"
pip install flash-attn==2.8.3 --no-build-isolation

git clone https://github.com/dmitrymailk/cut-cross-entropy.git
cd cut-cross-entropy && pip install -e .

git clone https://github.com/dmitrymailk/lm-evaluation-harness.git
cd lm-evaluation-harness/ && pip install -e .