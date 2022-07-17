import json
import datetime
import pprint
import os

def search_minmax_events(loaded_tweet_list):
    eventtimerdict = {}

    for tweet in loaded_tweet_list:
        loaded_time = datetime.datetime.strftime(datetime.datetime.strptime(tweet[3],'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        # loaded_time.year = 2000 # overwrite all to 2000. 
        # loaded_time.month
        # loaded_time.day
        # loaded_time.hour
        # loaded_time.minute
        # print(loaded_time)
        if not tweet[4] in eventtimerdict:
            eventtimerdict[tweet[4]] = {"min":loaded_time,"max":loaded_time}
            continue
        if eventtimerdict[tweet[4]]["min"]>loaded_time:
            eventtimerdict[tweet[4]]["min"] = loaded_time
        if eventtimerdict[tweet[4]]["max"]<loaded_time:
            eventtimerdict[tweet[4]]["max"] = loaded_time

    # pprint.pprint(eventtimerdict)
    return eventtimerdict
    {
    'charliehebdo-all-rnr-threads':   # no need to map month.
        {'max': '2015-01-22 19:23:53',
        'min': '2015-01-07 11:06:08'},
    'ebola-essien-all-rnr-threads':  # map -9
        {'max': '2014-10-15 08:06:25',
        'min': '2014-10-12 14:44:23'},
    'ferguson-all-rnr-threads':  # map -7
        {'max': '2014-09-14 16:21:57',
        'min': '2014-08-09 22:33:06'},
    'germanwings-crash-all-rnr-threads':  # map -2
        {'max': '2015-04-01 20:12:15',
       'min': '2015-03-24 10:37:41'},
    'gurlitt-all-rnr-threads': # map -10
        {'max': '2014-11-24 13:12:51',
         'min': '2014-11-20 01:20:18'},
    'ottawashooting-all-rnr-threads':  # map -9
        {'max': '2014-10-26 01:52:24',
        'min': '2014-10-22 13:55:50'},
    'prince-toronto-all-rnr-threads':  # map -10
        {'max': '2014-11-07 02:45:25',
        'min': '2014-11-03 22:29:15'},
    'putinmissing-all-rnr-threads': # map -2
        {'max': '2015-03-22 03:06:09',
        'min': '2015-03-13 00:04:02'},
    'sydneysiege-all-rnr-threads': # map -11
        {'max': '2014-12-18 22:07:45',
        'min': '2014-12-14 23:02:38'}
    }
    

def load_and_arrange(targetfile):
    with open(targetfile,"r",encoding="utf-8") as dumpfile:
        loaded_tweet_list = json.load(dumpfile)
    
    # search_minmax_events(loaded_tweet_list)
    preconfigured_dict = {
    'charliehebdo-all-rnr-threads': 0,
    'ebola-essien-all-rnr-threads': -9,
    'ferguson-all-rnr-threads': -7,
    'germanwings-crash-all-rnr-threads': -2,
    'gurlitt-all-rnr-threads': -10,
    'ottawashooting-all-rnr-threads': -9,
    'prince-toronto-all-rnr-threads': -10,
    'putinmissing-all-rnr-threads': -2,
    'sydneysiege-all-rnr-threads': -11}
    reloaded_tweets_event = {}
    reloaded_tweets = []
    # because all items are within the same year, there are less complications.
    for tweet in loaded_tweet_list:
        loaded_time = datetime.datetime.strftime(datetime.datetime.strptime(tweet[3],'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        # print(loaded_time)
        date = loaded_time.split()[0].split("-")
        date[0] = "2015"
        adjuster = preconfigured_dict[tweet[4]]
        date[1] = "{:02d}".format(int(date[1]) + adjuster)
        date = "-".join(date)+ " " + loaded_time.split()[1]
        newly_loaded_time = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
        # print(newly_loaded_time)
        # example of loading it into a datetime
        if not tweet[4].split("-")[0] in reloaded_tweets_event:
            reloaded_tweets_event[tweet[4].split("-")[0]] = []
        
        # all items are now mapped between the range of 2015 Jan onwards. 
        # The furthest tweet from 2015 Jan's first tweet would be FEB 14...
        reloaded_tweets_event[tweet[4].split("-")[0]].append((tweet[0],tweet[1],tweet[2],date,tweet[4],tweet[5]))
        reloaded_tweets.append((tweet[0],tweet[1],tweet[2],date,tweet[4],tweet[5]))
    
    reloaded_tweets.sort(key=lambda z:z[4])
    for event in reloaded_tweets_event:
        reloaded_tweets_event[event].sort(key=lambda z:z[4])
    # pprint.pprint(reloaded_tweets)
    
    # reloaded_tweets = the arranged tweets in order of the time in which they appear, ready for streaming.
    total_length = len(reloaded_tweets)
    k = int(total_length/5)
    return reloaded_tweets, reloaded_tweets_event, ((0,k),(k,2*k),(2*k,3*k),(3*k,4*k),(4*k,total_length))

if __name__=="__main__":
    if not "tweetwindow4.txt" in os.listdir(): # recreate all windows!
        loaded_tweets, loaded_event_split,index_targets = load_and_arrange("complete_tweet_list.json")
        # print(index_targets)
        for i in range(len(index_targets)):
            with open("tweetwindow_"+str(i)+".txt","w",encoding="utf-8") as dumpfile:
                json.dump(loaded_tweets[index_targets[i][0]:index_targets[i][0]],dumpfile,indent=4)
        with open("timed_event_splits.txt") as dumpfile:
            json.dump(loaded_event_split,dumpfile,indent = 4)