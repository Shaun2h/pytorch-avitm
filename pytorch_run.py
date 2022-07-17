import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.cuda
from pprint import pprint, pformat
import random
import pickle
import argparse
import json
import os
import math
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from pytorch_model import ProdLDA, Hypernet_LDA, Hypernet_LDA_RNN
from pytorch_visualize import *
from dataset_classes import dataset_class_PHEME_direct,dataset_class_PHEME_root_only_tree_inp
from torch.utils.data import Dataset, DataLoader
from PHEME_DATASET.pheme_dataset_loader_time import load_and_arrange


associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'comp ': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh', 
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'car  ': ['wheel', 'tire'],
    'polit': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'sport': ['coach', 'hitter', 'pitch'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
}
# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--en1-units',        type=int,   default=100)
# parser.add_argument('-s', '--en2-units',        type=int,   default=100)
# parser.add_argument('-t', '--num-topic',        type=int,   default=50)
# parser.add_argument('-b', '--batch-size',       type=int,   default=200)
# parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
# parser.add_argument('-r', '--learning-rate',    type=float, default=0.002)
# parser.add_argument('-m', '--momentum',         type=float, default=0.99)
# parser.add_argument('-e', '--num-epoch',        type=int,   default=80)
# parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
# parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
# parser.add_argument('--start',                  action='store_true')        # start training at invocation
# parser.add_argument('--nogpu',                  action='store_true')        # do not use GPU acceleration

# args = parser.parse_args()


# default to use GPU, but have to check if GPU exists
# if not args.nogpu:
    # if torch.cuda.device_count() == 0:
        # args.nogpu = True

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)


def make_data():
    dataset_tr = 'data/20news_clean/train.txt.npy'
    data_tr = np.load(dataset_tr,allow_pickle=True,encoding="latin1")
    dataset_te = 'data/20news_clean/test.txt.npy'
    data_te = np.load(dataset_te,allow_pickle=True,encoding="latin1")
    vocab = 'data/20news_clean/vocab.pkl'
    vocab = pickle.loads(open(vocab,'r').read().encode()) # gabu gabu
    vocab_size = len(vocab)
    # print(vocab)
    # --------------convert to one-hot representation------------------
    print('Converting data to one-hot representation')
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print('Data Loaded')
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    #--------------make tensor datasets-------------------------------
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    # if not args.nogpu:
        # tensor_tr = tensor_tr.cuda()
        # tensor_te = tensor_te.cuda()
    return data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size



def train(encoder_tokenizer,dataloader,backprop,settings,descriptor,model_namer,dosave=True):
    
    class nothing: # do... nothing with a with statement.
        def __init__(self):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exception_type, exception_value, traceback):
            pass
    
    if backprop:
        grad_decider = nothing
        # print("Using Grad")
        model.train()                   # switch to training mode

    else:
        grad_decider = torch.no_grad
        model.eval()

        # print("No Grad.")
    with grad_decider():
        # all_indices = torch.randperm(tensor_tr.size(0)).split(settings["batch_size"])
        loss_epoch = 0.0
        for _,(tokenized_data,label,idlist,raw_data) in enumerate(dataloader):
            # torch.set_printoptions(threshold=100000)
            # print(tokenized_data[0])
            # print(tokenized_data[0].shape)
            # print(torch.sum(tokenized_data[0],1))
            optimizer.zero_grad()       # clear previous gradients
            # batch_indices = batch_indices.to(settings["device"]) # i'm not sure why this is needed in the original code.
            # print(batch_indices)
            
            input_data = Variable(tokenized_data).to(settings["device"])
            if type(model)==Hypernet_LDA_RNN:
                recon, loss, _ = model(input_data, num_topic=random.randint(4,settings["num_topic"]),compute_loss=True)
            else:
                recon, loss, _ = model(input_data, compute_loss=True)
            
            # optimize if applicable
            if backprop:
                loss.backward()             # backprop
                optimizer.step()            # update parameters
            # report
            loss_epoch += loss.data[0]    # add loss to loss_epoch
            
        if epoch % 5 == 0:
            print('Epoch {}, {} loss={}'.format(epoch,descriptor, loss_epoch / len(dataloader.dataset)))
        if dosave:
            torch.save(model.state_dict(), model_namer+"_"+str(epoch)+".torch")


def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.items():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics

def print_top_words(beta, feature_names, n_top_words=50):
    outputstr = ""
    outputstr += '---------------Printing the Topics------------------'
    outputstr += "\n"
    for i in range(len(beta)):
        line = " ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        topics = identify_topic_in_line(line)
        # print('\n\n'.join(topics))
        outputstr +=str(line.split())
        outputstr += "\n"
    outputstr += '---------------End of Topics------------------' 
    outputstr += "\n"
    return outputstr

def print_perp(model,test_dataloader, verbose,num_topics=10):
    # oddly enough, the "counts" in the code we reference actually points at the input dimension's counts.
    cost=[]
    model.eval()                        # switch to testing mode
    losslist = []
    counts = 0       # we actually reference this in the event split with a proper explanation as to the ^512.
    clusterdicts = {}
    with torch.no_grad():
        for _,(tokenized_data,label,idlist,raw_data) in enumerate(test_dataloader):
            input = Variable(tokenized_data)
            counts = torch.sum(tokenized_data,dim=1)
            if type(model)==Hypernet_LDA_RNN:
                recon, loss, class_probs = model(input.to(settings_dict["device"]), num_topics,compute_loss=True, avg_loss=False)
            else:
                recon, loss, class_probs = model(input.to(settings_dict["device"]), compute_loss=True, avg_loss=False)
            _, selected_clusters = torch.max(class_probs,dim=1)
            # print(selected_clusters.shape)
            for specified_id in range(len(idlist)):
                _, rootid, selfid = test_dataloader.dataset.backref(idlist[specified_id])
                # print(selfid)
                # print(rootid)
                clusterdicts[rootid] = int(selected_clusters[specified_id])
                
            losslist.append(loss.data/counts)
    # print(torch.cat(losslist,dim=0).shape)
    avg = torch.cat(losslist,dim=0).mean()
    if verbose:
        print('The approximated final perplexity is: ', math.exp(avg))
    return clusterdicts


def print_perp_eventsplit(model,test_dataloader,eventnames, verbose):
    cost=[]
    model.eval()                        # switch to testing mode
    loss = None
    reverseeventnames = {}
    loss_saver_event = {}
    clusterdicts = {}
    print("-"*50)
    # print(eventnames)
    for event in eventnames:
        for rootid in eventnames[event]:
            reverseeventnames[rootid] = event
        loss_saver_event[event] = [[],0]
    counts = 0 # technically, it's the input size...
    # basically the counts is the number of.. tokens present in the distribution. 
    # we pad to 512.
    # if we removed the [PAD] option somehow it wouldn't be 512 per entry, but here we are...
    # a real vocab count would be different depending on sentence length and other things. but we have pad to fill to 512.
    # By removing all the [PAD] tokens, counts needs to be counted manually.
    
    with torch.no_grad():
        for _,(tokenized_data,label,idlist,raw_data) in enumerate(test_dataloader):
            input = Variable(tokenized_data)
            if type(model)==Hypernet_LDA_RNN:
                recon, loss, class_probs = model(input.to(settings_dict["device"]), len(list(eventnames.keys())),compute_loss=True, avg_loss=False)
            else:
                recon, loss, class_probs = model(input.to(settings_dict["device"]), compute_loss=True, avg_loss=False)
            counts = torch.sum(tokenized_data,dim=1)
            _, selected_clusters  = torch.max(class_probs,dim=1)

            # print(loss.shape)
            # print(loss)
            # print(idlist)
            for specified_id in range(len(idlist)):
                _, rootid,selfid = test_dataloader.dataset.backref(idlist[specified_id])
                # print(reverseeventnames[rootid],rootid)
                clusterdicts[int(selfid)] = int(selected_clusters[specified_id]) # pick out the predicted value for this particular tweet id and store.
                loss_saver_event[reverseeventnames[str(rootid)]][0].append(loss.data/counts) # store the loss for this event.
                loss_saver_event[reverseeventnames[str(rootid)]][1] = loss_saver_event[reverseeventnames[str(rootid)]][1] + 1 
                
                
            if loss!=None:
                loss += loss.data
            else:
                loss = loss.data

    # avg = (loss.cpu() / (counts)).mean()
    completed_loss = []
    for event in loss_saver_event:
        if loss_saver_event[event][0]:
            loss_saver_event[event].append(torch.cat(loss_saver_event[event][0],dim=0).mean())
            completed_loss.extend(loss_saver_event[event][0])
        else:
            loss_saver_event[event].append(torch.tensor([1]))
            completed_loss.extend(torch.tensor([1]))
    if verbose:
        for event in loss_saver_event:
            print(event," - Total loss", float(torch.sum(loss)), " - Total Tweets in event Test: ", loss_saver_event[event][1], " - Average Loss: ",loss_saver_event[event][2]/loss_saver_event[event][1])
            try:
                print(" - Perplexity: ", math.exp(loss_saver_event[event][2]))
            except OverflowError:
                print("Perplexity is too high to calculate for this event. Perform an exponent of the average loss reported above.")
        print("The total accumulated loss across all events was:",loss.cpu().mean())
        print("The total number of tweets tested Across all events was:", int(torch.sum(counts)))
        print('The approximated Average perplexity across all events is: ', math.exp(torch.cat(completed_loss,dim=0).mean()))
    return clusterdicts

def visualize():
    global recon
    input = Variable(tensor_te[:10])
    
    register_vis_hooks(model)
    recon = model(input.to(settings_dict["device"]), compute_loss=False)
    remove_vis_hooks()
    save_visualization('pytorch_model', 'png')

def encode_strings(encoder_model,sentences):
    outputs = encoder_model(**sentences)
    pooler_output  = outputs.pooler_output.reshape(-1) # flatten.
    return pooler_output

if __name__=='__main__':
    pheme_directory = "PHEME_DATASET"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pure_random = False # 4th in line
    event_splits = False # pick one. 
    timer_events_split = False
    timer_sequential = True
    
    
    
    
    if int(pure_random)+int(event_splits)+ int(timer_events_split)+int(timer_sequential)>1:
        print("Pick one of the four, not more.")
        quit()
    elif int(pure_random)+int(event_splits)+ int(timer_events_split)+int(timer_sequential)==0:
        print("Pick one of them, not less.")
        quit()
    
    encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab = encoder_tokenizer.get_vocab()

    overall_name = "Overall_event_splits"   
    if not "eventsplit_eventshuffle.json" in os.listdir(pheme_directory):
        with open(os.path.join(pheme_directory,"Eventsplit_details.txt"),"r",encoding="utf-8") as dumpfile:
            event_indexes = json.load(dumpfile)
        for event in event_indexes:
            random.shuffle(event_indexes[event])
        with open(os.path.join(pheme_directory,"eventsplit_eventshuffle.json"),"w",encoding="utf-8") as dumpfile:
            json.dump(event_indexes,dumpfile,indent=4)
        
    else:
        with open(os.path.join(pheme_directory,"eventsplit_eventshuffle.json"),"r",encoding="utf-8") as dumpfile:
            event_indexes = json.load(dumpfile)

        
        
    settings_dict = {
        "inp_dim":1995,
        "en1": 32,
        "en2": 32,
        "num_topic": 10,
        "init_mult": 1.0,
        "variance": 0.995,
        "optimizer": "Adam",
        "learning_rate":0.002,
        "momentum": 0.99,
        "device": device,
        "num_epoch":20,
        "batch_size":200
    }
    loaded_tweets, loaded_event_split, index_targets = load_and_arrange(os.path.join(pheme_directory,"complete_tweet_list.json"))
    labeldict = {'charliehebdo': 0, 'ebola': 1, 'ferguson': 2, 'germanwings': 3, 'gurlitt': 4, 'ottawashooting': 5, 'prince': 6, 'putinmissing': 7, 'sydneysiege': 8}
        
    if pure_random:
        # random split.
        overall_name = "Pure_random_splits"
        with open(os.path.join(pheme_directory,"pheme_traintest_splits.json"),"r",encoding="utf-8") as dumpfile:
            traintestindexes = json.load(dumpfile)
            traintestindexes = traintestindexes[0]
        trainidx = traintestindexes[0]
        testidx = traintestindexes[1]
        
        trainloads = []
        testloads = []
        for checked_tweet in loaded_tweets:
            # print(checked_tweet)
            if str(checked_tweet[-1]) in trainidx: # check the root tweet
                trainloads.append(checked_tweet)
            elif str(checked_tweet[-1]) in testidx:
                testloads.append(checked_tweet)
                
                
        
        train_dataset = dataset_class_PHEME_direct(trainloads,labeldict,encoder_tokenizer, device,"Train: ")
        test_dataset = dataset_class_PHEME_direct(testloads,labeldict,encoder_tokenizer, device,"Test: ")
        print("Done loading Datasets")
        topwords_namer = "pure_random_topwords.txt" # where to save topwords.


        
    elif event_splits:
        
        trainidx = []
        testidx = []
        for event in event_indexes:
            test_idx_event = event_indexes[event][int(len(event_indexes[event])/5*4):] # last 20%
            train_idx_event = event_indexes[event][:int(len(event_indexes[event])/5*4)] # first 80%
            trainidx.extend(train_idx_event)
            testidx.extend(test_idx_event)
        

        train_dataset = dataset_class_PHEME_root_only_tree_inp(trainidx, encoder_tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"),"Train: ")
        test_dataset = dataset_class_PHEME_root_only_tree_inp(testidx, encoder_tokenizer, device, os.path.join(pheme_directory,"phemethreaddump.json"),"Test: ")
        print("Done loading Datasets")
        topwords_namer = "event_split_topwords.txt" # where to save topwords.
        
        
    elif timer_events_split:
        overall_name = "Event_Sequential_timed"
        overalltrain_list = []
        overalltest_list = []
        for eventname in loaded_event_split:
            breaker = int(len(loaded_event_split[eventname])/5*4) # 80% train Change here to change ratio. (take first 20%)
            trainpart = loaded_event_split[eventname][:breaker]
            testpart = loaded_event_split[eventname][breaker:]
            overalltrain_list.extend(trainpart)
            overalltest_list.extend(testpart)
            
        train_dataset = dataset_class_PHEME_direct(overalltrain_list,labeldict,encoder_tokenizer, device,"Train: ")
        test_dataset = dataset_class_PHEME_direct(overalltest_list,labeldict,encoder_tokenizer, device,"Test: ")
        topwords_namer = "timer_event_split_topwords.txt" # where to save topwords.
        
        
    elif timer_sequential:
        overall_name = "Sequential_timed"
        
        train_dataset = dataset_class_PHEME_direct(loaded_tweets[:index_targets[0][1]],labeldict,encoder_tokenizer, device,"Train: ")
        test_dataset = dataset_class_PHEME_direct(loaded_tweets[index_targets[0][1]:],labeldict,encoder_tokenizer, device,"Test: ")
        topwords_namer = "timer_mixed_topwords.txt" # where to save topwords.
    
    

    
    
    # encoder_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    
    # inputs = encoder_tokenizer("some sample sentence here!", return_tensors="pt")
    # outputs = encoder_model(**inputs)
    # pooler_output  = outputs.pooler_output 
    # settings_dict["num_input"]= data_tr.shape[1]
    # settings_dict["num_input"]= pooler_output.shape[1]
    
    inp_size = max(list({vocab[z]:z for z in vocab}.keys()))
    settings_dict["num_input"] = inp_size - 1  # ignore PAD.
    

    with open(os.path.join(pheme_directory,"clusterlist.json"),"r",encoding="utf-8") as clusterfile:
        full_clusterlist = json.load(clusterfile)
        
    dua_map = {}
    counter = 0
    for cluster in full_clusterlist: # obtain the cluster to cluster number targets.
        for targetid in cluster[0]:
            dua_map[targetid] = counter
        counter+=1

    
    train_dataloader = DataLoader(train_dataset, batch_size=settings_dict["batch_size"], shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=settings_dict["batch_size"], shuffle=True, num_workers=0)    
    # model = Hypernet_LDA(settings_dict)
    # model = ProdLDA(settings_dict)
    model = Hypernet_LDA_RNN(settings_dict,device)

    model = model.to(device)
    
    with open(os.path.join(pheme_directory,"Eventsplit_details.txt"),"r",encoding="utf-8") as roottweetfile:
        event_root_list = json.load(roottweetfile)


    if settings_dict["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), settings_dict["learning_rate"], betas=(settings_dict["momentum"], 0.999))
    elif settings_dict["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), settings_dict["learning_rate"], momentum=settings_dict["momentum"])
    for epoch in range(settings_dict["num_epoch"]):
        
        train(encoder_tokenizer,train_dataloader,True,settings_dict,"-Train-",overall_name)
        train(encoder_tokenizer,test_dataloader,False,settings_dict,"-Test-",overall_name,False) # don't save model this time. This is test.
        break
        
    if type(model)==Hypernet_LDA_RNN:
        emb = model.decoderRNN_out_latest.data.cpu().numpy().T[:,1,:].reshape(-1,settings_dict["num_input"]) 
        # unlike the original, we customise the embed PER sample. as a result we have <input_size, batch_size,topic num>
        # arguably, the general decoder outputs should still be similarish.
        # that is to say while the decoder weights change from sample to sample, grabbing a random sample's decoder should still allow
        # a good enough average view of a topic.
        # print("Embed Shape:",emb.shape)
    else:
        emb = model.decoder.weight.data.cpu().numpy().T
        # print("Embed Shape:",emb.shape)
    outputstr = print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0][1:])
    with open(os.path.join(pheme_directory,topwords_namer),"w",encoding="utf-8") as topwordsdump:
        topwordsdump.write(outputstr)
    
    if pure_random:
        tests_predictedclusters = print_perp(model, test_dataloader, verbose=True)
        trains_predictedclusters = print_perp(model, train_dataloader, verbose=False)
    elif event_splits or timer_events_split or timer_sequential:
        tests_predictedclusters = print_perp_eventsplit(model, test_dataloader, event_indexes, verbose=True)
        trains_predictedclusters = print_perp_eventsplit(model, train_dataloader, event_indexes, verbose=False)
    


    predictedclusters = {}
    predictedclusters.update(trains_predictedclusters)
    predictedclusters.update(tests_predictedclusters)
    
    event_comparison = {}
    
    overall_correct_count = 0
    overall_wrong_count = 0
    
    for event in event_root_list:
        event_comparison[event] = [0,0]
        
    touched_list = []
    unpredictedroots = []
    for tweetid in predictedclusters:
        clusternumber = dua_map[str(tweetid)] # obtain the cluster number
        
        if clusternumber in touched_list: # skip if the cluster has already been investigated.
            continue
            
        member_list = full_clusterlist[clusternumber][0] # extract members in the cluster.
        touched_list.append(clusternumber) # record which clusters have been investigated.
        root_tweet = member_list[0]
        # print(predictedclusters)
        # print(root_tweet)
        # print(member_list)
        
        try:
            root_cluster = predictedclusters[int(root_tweet)]
        except KeyError:
            unpredictedroots.append(root_tweet)
            continue

        source_event = None
        for targetevent in event_root_list:  # log the event it belongs to.
            if root_tweet in event_root_list[targetevent]:
                source_event = targetevent
        

        for member in member_list:
            if not int(member) in predictedclusters:
                continue # dropped due to lack of connection to root tweet... might edit out later in creation of data?
            # print(predictedclusters[int(member)])
            # print(root_cluster)
            if predictedclusters[int(member)] == root_cluster:
                event_comparison[source_event][0] = event_comparison[source_event][0] + 1 # event wise correct.
                overall_correct_count+=1
            else:
                event_comparison[source_event][1] = event_comparison[source_event][1] + 1 # event wise correct.
                overall_wrong_count+=1 

    # print(overall_correct_count,overall_wrong_count)
    print("Overall Percentage Parent Child Match:", overall_correct_count/(overall_wrong_count+overall_correct_count))
    print("Overall Correct Parent Child Match:",overall_correct_count)
    print("Overall Wrong Parent Child Match:",overall_wrong_count)
    print("-"*50)
    for event in event_comparison:
        print("Overall Correct in", event,":", event_comparison[source_event][0])
        print("Overall Wrong in", event,":", event_comparison[source_event][1])
        print("Overall Percentage Parent Child Match for",event,":",event_comparison[source_event][0]/(event_comparison[source_event][0]+event_comparison[source_event][1]))
        print("-"*50)
    print("Unpredicted Roots:",unpredictedroots)
    # visualize()


