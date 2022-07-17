import json
import os
import numpy as np


# Run with all-rnr-annotated-threads in the background.
# this serves to generate the REQUIRED file for pheme 
# please understand there are a lot of sparse trees so this is a highly DIFFICULT dataset if you're gcning it
# Outputs: phemethreaddump.json, labelsplits.txt, eventsplit_details.txt
# phemethreaddump: AdjacencyMatrix+ Threads
# labelsplits: The tree's root node + it's associated label. Event not saved.
# eventsplitddetails: saves which root node belongs to which event


def traversal(ref_dict,currenttarget):
    returnval = []
    for i in ref_dict[currenttarget]:
        returnval.extend(traversal(ref_dict,i))
    for item in returnval:
        item.insert(0,currenttarget)
    returnval.append([currenttarget])
    return returnval

FORCE_ROOT_CONNECTION=False

if not os.path.exists("phemethreaddump.json") or not os.path.exists("labelsplits.txt") or not os.path.exists("Eventsplit_details.txt") or not os.path.exists("complete_tweet_list.json") or not os.path.exists("clusterlist.json"):
    if not os.path.exists("phemethreaddump.json"):
        pheme_root = "all-rnr-annotated-threads" # POINT ME
        all_tweets_list = []
        eventlist = os.listdir(pheme_root)
        allthreads = []
        full_clusterlist = []
        print("beginning to load pheme dataset/setup files since they weren't done.")
        for event in eventlist:
            if "." ==event[0]:
                continue
            eventname = event.replace("-all-rnr-threads","")
            for classification in ["non-rumours","rumours"]:
                if classification == "non-rumours":
                    rootlabel = (0,"non-rumours",eventname)
                else:
                    rootlabel = (1,"rumour",eventname)
                    
                for somethread in os.listdir(os.path.join(pheme_root,event,classification)):
                    clusterlist = []

                    if "." ==somethread[0]:
                        continue
                    sourcetweetfile = os.path.join(pheme_root,event,classification,somethread,"source-tweets",somethread+".json")
                    approval_set = set()
                    with open(sourcetweetfile,"r",encoding="utf-8") as opened_sourcetweetfile:
                        sourcedict = json.load(opened_sourcetweetfile)
                    sourcetweet = (sourcedict["text"], sourcedict["id_str"], sourcedict["id"], sourcedict["created_at"],event)
                    all_tweets_list.append(sourcetweet+(sourcedict["id"],))
                    # ["tweettext","tweetid","authid"]
                    threadtextlist = [sourcetweet]
                    tree = {sourcetweet[1]:[]}
                    clusterlist.append(sourcetweet[1])
                    reactions = os.listdir(os.path.join(pheme_root,event,classification,somethread,"reactions"))
                    for reactionfilename in reactions:
                        if "." ==reactionfilename[0]:
                            continue
                        reactionsfilepath = os.path.join(pheme_root,event,classification,somethread,"reactions",reactionfilename)
                        with open(reactionsfilepath,"r",encoding="utf-8") as opened_reactiontweetfile:
                            reactiondict = json.load(opened_reactiontweetfile)
                            reactiontweet = (reactiondict["text"], reactiondict["id_str"], reactiondict["id"], reactiondict["created_at"],event) 
                            # ["tweettext","tweetid","authid"]
                            if reactiontweet[1]==sourcetweet[1]: # this is an actual pheme problem that NEEDS TO STOP
                                print("There's a dupe for a reaction/root node. Pheme specific problem.")
                                print(reactiondict["text"]) # WHY IS A SOURCE TWEET IN THE REACTIONS FOLDER???
                                print(sourcedict["text"])
                                continue 
                            
                            
                            all_tweets_list.append(reactiontweet+(sourcedict["id"],))   # We don't append reaction tweets normally.
                            
                            clusterlist.append(reactiontweet[1])
                            threadtextlist.append(reactiontweet)
                            replytarget = reactiondict["in_reply_to_status_id"]
                            if not reactiondict["id_str"] in tree: # if self isn't in tree.
                                tree[reactiondict["id_str"]] = [] # place self into treedict
                            
                            if str(replytarget)+".json" in reactions or str(replytarget) in tree:
                                # print(replytarget)
                                # print(reactions)
                                if not str(replytarget) in tree:
                                    tree[str(replytarget)] = [] # if the response target hasn't been added but is a valid tweet in the dataset,
                                tree[str(replytarget)].append(reactionfilename.replace(".json",""))
                        if FORCE_ROOT_CONNECTION:
                            variants = traversal(tree,sourcetweet[1]) # traverse for ALL POSSIBLE rootwalks
                            for treewalk in variants:
                                for nodename in treewalk:
                                    approval_set.add(nodename)
                            allowed_list = list(approval_set)
                            for treetarget in list(tree.keys()):
                                if not treetarget in allowed_list:
                                    del tree[treetarget]
                            finalthreadlist = []
                            for i in threadtextlist:
                                if i[1] in allowed_list:
                                    finalthreadlist.append(i)
                            threadtextlist = finalthreadlist
                                
                    full_clusterlist.append([clusterlist, event])
                    allthreads.append([threadtextlist,tree,rootlabel,sourcedict["id_str"]])
                    
        print("Parsed all files.")
        with open("phemethreaddump.json","w",encoding="utf-8") as dumpfile:
            json.dump(allthreads,dumpfile,indent=4)
        print("Thread dump completed (you can even delete the dataset now! wow!)")
        with open("complete_tweet_list.json","w",encoding="utf-8") as dumpfile:
            json.dump(all_tweets_list,dumpfile,indent=4)
            
        with open("clusterlist.json","w",encoding="utf-8") as clusterfile:
            json.dump(full_clusterlist,clusterfile,indent=4)
            
        print("Thread dump completed (you can even delete the dataset now! wow!)")
    if not os.path.exists("PHEME_labelsplits.txt"):        
        with open("phemethreaddump.json","r",encoding="utf-8") as dumpfile: #
            allthreads = json.load(dumpfile)
        with open("PHEME_labelsplits.txt","w") as labelfile:

            for thread in allthreads:
                threadtextlist,tree,rootlabel,source_id = thread
                labelfile.write(str(source_id)+" "+str(rootlabel)+"\n")
    
    if not os.path.exists("Eventsplit_details.txt"): # save all event ids separately for use as folds..
        with open("phemethreaddump.json","r",encoding="utf-8") as dumpfile: # YES LEAVE IT IN ***** # edit this in later lol
            allthreads = json.load(dumpfile)
        with open("Eventsplit_details.txt","w") as eventsplitfile:
            eventsplits = {}
            for thread in allthreads:
                threadtextlist,tree,rootlabel,source_id = thread
                # rootlabel = (0,"non-rumours",eventname)
                if not rootlabel[2] in eventsplits:
                    eventsplits[rootlabel[2]] = []
                eventsplits[rootlabel[2]].append(source_id)
            json.dump(eventsplits,eventsplitfile,indent=4)
                    