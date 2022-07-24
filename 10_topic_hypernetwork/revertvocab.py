import os
import ast
from transformers import BertTokenizer, BertModel


encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab = encoder_tokenizer.get_vocab()
inv_vocab = {i:k for k,i in vocab.items()}
# print(inv_vocab)
for directory in os.listdir():
    if not os.path.isdir(directory):
        continue
    for item in os.listdir(directory):
        if not "topwords" in item or "fixed" in item:
            continue
        allouts = []
        with open(os.path.join(directory,item),"r",encoding="utf-8") as a:
            for line in a:
                if line:
                    if line[0]!="-":
                        loaded_list = ast.literal_eval(line)
                        outylist = []
                        for i in loaded_list:
                            outylist.append(inv_vocab[vocab[i]+2])
                        allouts.append(outylist)
        with open(os.path.join(directory,item[:-4]+"_fixed.txt"),"w",encoding="utf-8") as out:
            for outer in allouts:
                out.write(outer.__str__())
                out.write("\n")
                            
                        