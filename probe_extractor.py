import os
import sys
import json
import torch
import pickle
import collections

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BartTokenizer, AutoModelForSeq2SeqLM, BartConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

class InputExample(object):
    def __init__(self, text, unique_id):
        self.text = text
        self.unique_id = unique_id

class InputFeatures(object):
    def __init__(self, tokens, unique_id, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.unique_id = unique_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def read_examples(text_file):
    examples = []
    unique_id = 0

    with open(text_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line: break

            text = line.strip().split('\t')[-1]
            examples.append(InputExample(text=text, unique_id=unique_id))
            unique_id += 1
    return examples  

def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        cand_tokens = tokenizer.tokenize(example.text)
        if len(cand_tokens) > seq_length - 2: 
            ## Account for [CLS] and [SEP] with "- 2"
            cand_tokens = cand_tokens[0:(seq_length - 2)] 

        tokens = []
        input_type_ids = []
        
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids  = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(InputFeatures(tokens=tokens, unique_id=example.unique_id, input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids))
    return features

def get_max_seq_length(samples, tokenizer):
    max_seq_len = -1
    for sample in samples:
        cand_tokens = tokenizer.tokenize((sample.text))
        cur_len = len(cand_tokens)
        if cur_len > max_seq_len:
            max_seq_len = cur_len

    # *************************************
    if max_seq_len > model_max_seq_length:
        max_seq_len = model_max_seq_length
    # *************************************

    return max_seq_len

def save_features(model, tokenizer, device):
    # convert data to ids
    examples = read_examples(text_dataset)
    features = convert_examples_to_features(examples=examples, seq_length=(get_max_seq_length(examples, tokenizer)), tokenizer=tokenizer)

    # extract and write features
    all_input_ids       = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask      = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_indices = torch.arange(all_input_ids.size(0), dtype=torch.long) # gives => tensor([0,1, 2, ... (num_samples - 1) ])
    eval_dataset        = TensorDataset(all_input_ids, all_input_mask, all_example_indices)
    eval_dataloader     = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batchsize)

    pbar = tqdm(total=len(examples)//batchsize)
    with open(json_features, "w") as writer:
        with torch.no_grad():
            for input_ids, input_mask, example_indices in eval_dataloader: # batch_sized input_ids, input_mask, example_indices tensor
                input_ids   = input_ids.to(device)    # batch_sized input_ids tensor
                input_mask  = input_mask.to(device)   # batch_sized input_mask tensor
                if "plbart" in model.__dict__["config"]._name_or_path:
                    all_outputs = model(input_ids=input_ids)#, token_type_ids=None, attention_mask=input_mask) 
                    enc_layers  = all_outputs.encoder_hidden_states 
                elif "codet5" in model.__dict__["config"]._name_or_path:
                    all_outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)#, token_type_ids=None, attention_mask=input_mask) 
                    enc_layers  = all_outputs.encoder_hidden_states                     
                else:
                    all_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask) 
                    enc_layers  = all_outputs.hidden_states          
                #print("***************************************************")
                #print(model_checkpoint, " => Num layers:", len(enc_layers))
                #print("***************************************************")

                for iter_index, example_index in enumerate(example_indices):
                    # for every feature in batch => tokens, input_ids, input_mask => features[example_index.item()]
                    feature     = features[example_index.item()] # example_indices are i,j,k, ... till batch_size
                    unique_id   = int(feature.unique_id)

                    all_output_features = []
                    for (token_index, token) in enumerate(feature.tokens):
                        all_layers = []
                        for layer_index in range(len(enc_layers)):
                            layer_output = enc_layers[int(layer_index)]  # layer   layer_index (#0, #1, #2 ... max_layers)
                            layer_feat_output = layer_output[iter_index] # feature iter_index 

                            layers = collections.OrderedDict()
                            layers["index"] = layer_index
                            layers["values"] = [round(hidden_unit.item(), 6) for hidden_unit in layer_feat_output[token_index]] # layer layer_index, feature iter_index, token token_index
                            all_layers.append(layers)

                        out_features = collections.OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_output_features.append(out_features)
                        break # if breaking only [CLS] token will be considered for classification

                    output_json = collections.OrderedDict()
                    output_json["linex_index"] = unique_id
                    output_json["features"] = all_output_features
                    writer.write(json.dumps(output_json) + "\n")

                pbar.update(1)
    pbar.close()
    print('written features to %s'%(json_features))


if __name__ == '__main__':

    task_codes    = ['CPT'] #['AST', 'CPX', 'CSC', 'IDF', 'IDT', 'JBL', 'JFT', 'JMB', 'LEN', 'MXN', 'NML', 'NMS', 'NPT', 'OCT', 'OCU', 'REA', 'SCK', 'SRI', 'SRK', 'TAN', 'TYP', 'VCT', 'VCU']
    shuffle_kinds = ['ORIG']
    label_counts  = ['200', '2k', '20k']

    model_checkpoints = {
        "BERT":          "bert-base-uncased", 
        "CodeBERT":      "microsoft/codebert-base",
        "CodeBERTa":     "huggingface/CodeBERTa-small-v1", 
        "GraphCodeBERT": "microsoft/graphcodebert-base",
        "CodeT5":        "Salesforce/codet5-base",
        "JavaBERT-mini": "anjandash/JavaBERT-mini",
        "PLBART-mtjava": "uclanlp/plbart-multi_task-java",
        "PLBART-large":  "uclanlp/plbart-large",
        #"GPT-J-6B":      "EleutherAI/gpt-j-6B",
    }

    model_max_seq_lengths = {
        "BERT":           512,
        "CodeBERT":       256, 
        "CodeBERTa":      512,
        "GraphCodeBERT":  512,
        "CodeT5":         512,
        "JavaBERT-mini":  512,
        "PLBART-mtjava":  1024,
        "PLBART-large":   1024,    
        #"GPT-J-6B":       2048,

    }


    for task_code in task_codes:
        for shuffle_kind in shuffle_kinds:
            for model_checkpoint in list(model_checkpoints.keys()):
                for label_count in label_counts:
                    print("********")
                    print(f"Processing for task >> {task_code} >> {shuffle_kind}:{model_checkpoint} for {label_count}")
                    print("********")

                    text_dataset  = sys.path[0] + '/data/datasets_'+ task_code +'/'+ task_code +'_'+ shuffle_kind +'_'+ label_count +'.txt'
                    json_features = sys.path[0] + '/data/datasets_'+ task_code +'/'+ shuffle_kind +'/'+ model_checkpoint +'_features_'+ label_count +'.json'

                    if not os.path.exists(json_features):
                        path = Path(json_features)
                        path.parent.mkdir(parents=True, exist_ok=True) 

                    # *******************************************************

                    modelname = model_checkpoints.get(model_checkpoint, None)
                    model_max_seq_length = model_max_seq_lengths.get(model_checkpoint, None)

                    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    batchsize = 1 #8 for 512 tokens 4 for 1024 tokens

                    if model_checkpoint == "BERT":

                        config    = BertConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=True, cache_dir="~/tmp")
                        model     = BertModel.from_pretrained(modelname, config=config, cache_dir="~/tmp")

                    elif model_checkpoint in ["CodeBERT", "CodeBERTa", "GraphCodeBERT"]:
                        
                        config    = RobertaConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = RobertaTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model     = RobertaForSequenceClassification.from_pretrained(modelname, config=config, cache_dir="~/tmp")

                    elif model_checkpoint in ["PLBART-mtjava", "PLBART-large", "CodeT5"]:

                        config    = AutoConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model     = AutoModelForSeq2SeqLM.from_pretrained(modelname, config=config, cache_dir="~/tmp") 

                    elif model_checkpoint in ["JavaBERT-mini"]:

                        config    = AutoConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model     = AutoModelForSequenceClassification.from_pretrained(modelname, config=config, cache_dir="~/tmp")   

                    elif model_checkpoint in ["GPT-J-6B"]:

                        config    = AutoConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model     = AutoModelForCausalLM.from_pretrained(modelname, config=config, cache_dir="~/tmp")                                                

                    print("-----")
                    print("Vocabulary  Size:\t", model.config.vocab_size)
                    print("Tokenizer Length:\t", len(tokenizer))
                    print("-----")
                    model.to(device)
                    model.eval()
                    save_features(model, tokenizer, device)
                    print("********")



