pip install -r requirements.txt
# cd ao && pip install -e . && cd ..
# cd cut-cross-entropy && pip install -e . && cd ..
# cd Liger-Kernel && pip install -e . && cd ..
# cd torchtune && pip install -e . && cd ..

pip install flash-attn==2.8.0.post2 --no-build-isolation

git clone https://github.com/dmitrymailk/cut-cross-entropy.git
cd cut-cross-entropy && pip install -e .

git clone https://github.com/dmitrymailk/lm-evaluation-harness.git
cd lm-evaluation-harness/ && pip install -e .