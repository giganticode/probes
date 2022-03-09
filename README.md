# Quickstart

1. First we will create a virtual environment with anaconda.   
If you do not have anaconda installed, please check the installation instructions [here](https://docs.anaconda.com/anaconda/install/index.html). 

   ```
      $ conda create --name probe_env python=3.8
   ```   
    
2. Next we will activate the virtual environment.

   ```
      $ conda activate probe_env
   ```

3. Once we are done with that, we will clone this github repository and `cd` into it.

   ```
      $ git clone https://github.com/giganticode/probes.git && cd probes/
   ```

4. Next we will install all the necessary packages in the *requirements.txt* file

   ```
      $ pip3 install -r requirements.txt
   ```

5. We will then begin generating input feature vectors from the pre-trained model weights. 

   ```
      $ CUDA_VISIBLE_DEVICES=0 python3 probe_extractor.py
   ```

6. Finally, we will use the pre-trained vectors to make some predictions.

   ```
      $ CUDA_VISIBLE_DEVICES=0 python3 probe_classifier.py > results.txt
   ```



Hurray! You're done! 🎉  
The final task accuracies of each model (layer by layer) will be saved in *results.txt* file. 

  *NOTE: In steps 5 & 6, we assume that you have access to a gpu with the cuda toolkit installed. If you are using a cpu instead, you can simply run the python scripts without the `CUDA_VISIBLE_DEVICES=0` part.*

---



## Model Performance by Tasks

### Code Length Prediction [![Generic badge](https://img.shields.io/badge/surface_information-blue.svg)]()

![Code Length Prediction Results](https://i.postimg.cc/v801rm54/Code-Length-Prediction-Results.png)

### AST Node Tagging Task [![Generic badge](https://img.shields.io/badge/syntactic_information-violet.svg)]()

![AST Node Tagging Results](https://i.postimg.cc/QtLgrdf8/AST-Node-Tagging-Results.png)

### Cyclomatic Complexity [![Generic badge](https://img.shields.io/badge/structural_information-darkgreen.svg)]()

![Cyclomatic Complexity Results](https://i.postimg.cc/02rzcqCc/Cyclomatic-Complexity-Results.png)

### Invalid Type Prediction [![Generic badge](https://img.shields.io/badge/semantic_information-orange.svg)]()

![Invalid Type Prediction Results](https://i.postimg.cc/kGYBqj7Q/Invalid-Type-Prediction-Results.png)



---



### Acknowledgements

This repository uses and is built upon the following works:

- [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) from huggingface
- [SentEval](https://github.com/facebookresearch/SentEval) from facebookresearch
- [bert-syntax](https://github.com/yoavg/bert-syntax) from yoavg
- [tpdn](https://github.com/tommccoy1/tpdn) from tommccoy1
- [rnn_agreement](https://github.com/TalLinzen/rnn_agreement) from TalLinzen
- [Chu-Liu-Edmonds](https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/mst/mst.ipynb) from bastings
- [interpret_bert](https://github.com/ganeshjawahar/interpret_bert) from ganeshjawahar



### License

[![MIT license](https://img.shields.io/badge/License-MIT-red.svg)](https://lbesson.mit-license.org/)

