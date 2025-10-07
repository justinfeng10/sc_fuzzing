import getopt
import os
import sys
import time
import random

import pandas as pd
from scipy.cluster import hierarchy

sys.path.append(r'..')

from SnR import Messenger
from Seed import Message, Seed


#  Global var
queue = []
restoreSeed = ''
outputfold = ''

def readBinarySeed(bin_path):
    """
    Read binary file and build a Seed containing a single Message.
    The binary is decoded with 'latin-1' to preserve raw 0-255 byte values
    and stored in raw["Content"] to match expectations from mutation/probe logic.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    with open(bin_path, "rb") as f:
        data = f.read()

    # decode with latin-1 so 1:1 mapping of byte → str char
    content_str = data.decode("latin-1")

    # Create Message
    msg = Message()

    # Manually add expected 'raw' and 'headers' attributes
    msg.raw = {"Content": content_str}
    msg.headers = ["Content"]  # Even if fake, needed by your write functions

    # Create Seed and attach message
    seed = Seed()
    seed.M = [msg]
    seed.R = []
    seed.PR = []
    seed.PS = []
    seed.PI = []
    seed.Snippet = []
    seed.ClusterList = []
    seed.isMutated = False

    return seed

def readBinInputFold(fold):
    seeds = []
    files = os.listdir(fold)
    for file in files:
        if file.endswith(".bin"):
            print("Loading binary file: ", os.path.join(fold, file))
            seeds.append(readBinarySeed(os.path.join(fold, file)))
    return seeds



# read the input file and store it as seed
def readInputFile(file):
    s = Seed()
    lines = []
    with open(file, 'r') as f:
        lines = f.read().split("\n")
    for i in range(0, len(lines)):
        # print(lines[i])
        if "========" in lines[i]:
            mes = Message()
            for j in range(i + 1, len(lines)):
                if "========" in lines[j]:
                    i = j
                    break
                if ":" in lines[j]:
                    mes.append(lines[j])
            s.append(mes)
    # s.display()
    return s


# read the input fold and store them as seeds
def readInputFold(fold):
    seeds = []
    files = os.listdir(fold)
    for file in files:
        print("Loading file: ", os.path.join(fold, file))
        seeds.append(readInputFile(os.path.join(fold, file)))
    return seeds


# Write the probe result that has been run into the output
def writeRecord(queue, fold):
    # TO DO: make compatible with bin format
    with open(os.path.join(fold, 'ProbeRecord.txt'), 'w') as f:
        for i in range(len(queue)):
            f.writelines("========Seed " + str(i) + "========\n")
            for j in range(len(queue[i].M)):

                f.writelines("Message Index-" + str(j) + "\n")  # write the message information
                for header in queue[i].M[j].headers:
                    f.writelines(header + ":" + queue[i].M[j].raw[header] + '\n')
                f.writelines("\n")

                f.writelines('Original Response' + "\n")  # write the original response
                f.writelines(queue[i].R[j])

                f.writelines('Probe Result:' + "\n")  # write the results of probe
                f.writelines('PI' + "\n")  # PI
                for n in queue[i].PI[j]:
                    f.write(str(n) + " ")
                f.writelines("\n")
                f.writelines('PR and PS' + "\n")
                for n in range(len(queue[i].PR[j])):
                    f.writelines("(" + str(n) + ") " + queue[i].PR[j][n])
                    f.writelines(str(queue[i].PS[j][n]) + "\n")
            f.writelines("\n")
            f.writelines("\n")
    return 0


# Read the probe results from the record, thus skip the probe process and directly start the mutation test.
def readRecordFile(file):
    queue = []
    with open(os.path.join(file), 'r') as f:
        lines = f.readlines()
        i = 0
        while i <= len(lines):
            if lines[i].startswith("========Seed"):
                seedStart = i + 1
                seedEnd = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[i].startswith("========Seed"):
                        seedEnd = j
                seed = Seed()
                index = seedStart

                while index <= seedEnd:

                    if lines[index].startswith('Message Index'):
                        message = Message()
                        responseStart = seedEnd
                        for j in range(index, seedEnd):
                            if lines[j].startswith('Original Response'):
                                responseStart = j
                                break
                        for line in lines[index + 1:responseStart - 1]:
                            message.append(line)
                        seed.M.append(message)
                        index = responseStart

                    if lines[index].startswith('Original Response'):
                        index = index + 1
                        seed.R.append(lines[index])

                    if lines[index].startswith('PI'):
                        index = index + 1
                        PIstr = lines[index]
                        PI = []
                        for n in PIstr.strip().split(' '):
                            PI.append(int(n))
                        seed.PI.append(PI)

                    if lines[index].startswith('PR and PS'):
                        index = index + 1
                        ends = seedEnd
                        PR = []
                        PS = []
                        for j in range(index, seedEnd):
                            if lines[j].startswith('Message Index'):
                                ends = j
                                break
                        for j in range(index, ends):
                            if lines[j].startswith("("):
                                PR.append(lines[j][3:])
                            elif lines[j][0].isdigit():
                                PS.append(float(lines[j].strip()))
                        seed.PR.append(PR)
                        seed.PS.append(PS)

                    index = index + 1

                i = index
                queue.append(seed)

            i = i + 1
    return queue


# Try to use the input given for a complete communication.
# The func is used to test whether the input meets the requirements or whether there are other problems
def dryRun(queue):
    global restoreSeed
    m = Messenger(restoreSeed)

    for i in range(len(queue)):
        result = m.DryRunSend(queue[i])
        if result is None:
            print(f"Dry run failed for seed {i}")
            return True  # Signals failure to caller
        queue[i] = result  # Safe: only update if result is good

    return False  # All succeeded


# Calculate the edit distance of two string   
def EditDistanceRecursive(str1, str2):
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)
    return edit[len(str1)][len(str2)]


def SimilarityScore(num_samples1, num_samples2):
    # Handle edge case if both are zero
    if num_samples1 == 0 and num_samples2 == 0:
        return 100.0  # identical
    
    max_val = max(num_samples1, num_samples2)
    diff = abs(num_samples1 - num_samples2)
    
    similarity = (1 - (diff / max_val)) * 100
    return round(similarity, 2)


# Use heuristics to detect the meaning of each byte in the message
def Probe(Seed):
    global restoreSeed

    print("*** Probe ")
    m = Messenger(restoreSeed)
    for index in range(len(Seed.M)):

        responsePool = []
        similarityScore = []
        probeResponseIndex = []

        print(Seed.M[index].raw["Content"].strip())  # test only
        # original message
        response1 = m.ProbeSend(Seed, index)  # send the probe message   ####### 
        time.sleep(1)
        response2 = m.ProbeSend(Seed, index)  # send the probe message twice

        responsePool.append(response1)
        similarityScore.append(SimilarityScore(response1, response2))

        # probe process
        # NOTE: doing -2 here because last two bytes fails
        for i in range(0, len(Seed.M[index].raw["Content"])-2):
            temp = Seed.M[index].raw["Content"]
            Seed.M[index].raw["Content"] = Seed.M[index].raw["Content"].strip()[:i] + Seed.M[index].raw[
                                                                                          "Content"].strip()[
                                                                                      i + 1:]  # delete ith byte
            response1 = m.ProbeSend(Seed, index)  # send the probe message   ####### 
            time.sleep(1)
            response2 = m.ProbeSend(Seed, index)  # send the probe message twice
            print(response1,end='')

            if responsePool:
                flag = True
                for j in range(0, len(responsePool)):
                    target = responsePool[j]
                    score = similarityScore[j]
                    c = SimilarityScore(target, response1)
                    if c >= score:
                        flag = False
                        probeResponseIndex.append(j)
                        print(str(j)+" ", end='') 
                        sys.stdout.flush()
                        break
                if flag:
                    responsePool.append(response1)
                    similarityScore.append(SimilarityScore(response1, response2))
                    probeResponseIndex.append(j + 1)
                    #print(j + 1)  # test only

            Seed.M[index].raw["Content"] = temp  # restore the message

        # STATS
        unique_categories = len(set(probeResponseIndex))
        print(f"[+] Message {index} has {unique_categories} unique probe response categories")

        Seed.PR.append(responsePool)
        Seed.PS.append(similarityScore)
        Seed.PI.append(probeResponseIndex)

    return Seed


def getFeature(response, score):
    # Return a simple 2D feature vector for clustering
    # Features could be the num_samples itself and its similarity score (or derived features)
    return [float(response), float(score)]

    '''
    feature = {}
    feature['a'] = 0
    feature['n'] = 0
    feature['s'] = 0
    length = len(response)
    score = score

    cur = ''
    pre = ''
    for i in range(len(response)):
        if response[i].isdigit():
            cur = 'n'
        elif response[i].isalpha():
            cur = 'a'
        else:
            cur = 's'

        if pre == '':
            pre = cur
        elif pre != cur:
            feature[pre] = feature[pre] + 1
        pre = cur

    feature[cur] = feature[cur] + 1

    # print(response)  # test only
    # print([feature['a'],feature['n'],feature['s'],length,score]) # test only

    return [feature['a'], feature['n'], feature['s'], length, score]
    '''

def formSnippets(pi, cluster, index):
    snippet = []
    for i in range(index):
        c1 = int(cluster[i][0])
        c2 = int(cluster[i][1])
        p = int(cluster[i][3])
        for j in range(len(pi)):
            if pi[j] == c1 or pi[j] == c2:
                pi[j] = p

    i = 0
    while i < len(pi)-1:
        j = i
        #print("i="+str(i)) # test only
        skip = True
        while j <= len(pi) and skip:
            j = j + 1
            #print("j=" + str(j))  # test only
            if pi[j] != pi[i]:
                snippet.append([i, j - 1])
                skip = False
            if j == len(pi)-1:
                snippet.append([i, j])
                skip = False
        i = j

    #print(pi)  # test only
    #print(snippet)   # test only

    return snippet


def interesting(oldSeed,index):
    global queue
    global restoreSeed
    m = Messenger(restoreSeed)

    
    print(oldSeed.M[index].raw["Content"])

    seed = Seed()
    seed.M = oldSeed.M
    seed = m.DryRunSend(seed)
    seed = Probe(seed)
    queue.append(seed)


def writeOutput(seed):
    global outputfold
    localtime = time.localtime(time.time())
    file = 'Crash-'+localtime+'.txt'

    with open(os.path.join(outputfold, file), 'w') as f:
        for i in range(len(seed)):
            f.writelines("Message Index-" + str(i) + "\n")  # write the message information
            for header in seed.M[i].headers:
                f.writelines(header + ":" + seed.M[i].raw[header] + '\n')
            f.writelines("\n")
    print("Found a crash @ "+localtime)
    sys.exit()


def responseHandle(seed, info):
    if info.startswith("#interesting"):
        print("~~Get Interesting in :")
        interesting(seed, int(info.split('-')[1]))
        return False
    if info.startswith("#error"):
        print("~~Something wrong with the target infomation (e.g. IP addresss or port)")
    if info.startswith("#crash"):
        print("OK")
        writeOutput(seed)
    return True


def SnippetMutate(seed, restoreSeed):
    
    m = Messenger(restoreSeed)

    for i in range(len(seed.M)):
        pool = seed.PR[i]
        poolIndex = seed.PI[i]
        similarityScores = seed.PS[i]

        featureList = []
        for j in range(len(pool)):
            featureList.append(getFeature(pool[j], similarityScores[j]))

        df = pd.DataFrame(featureList)
        cluster = hierarchy.linkage(df, method='average', metric='euclidean')
        #print("Cluster:")
        #print(cluster)
        #seed.display()

        seed.ClusterList.append(cluster)

        mutatedSnippet = []
        for index in range(len(cluster)):
            snippetsList = formSnippets(poolIndex, cluster, index)
            for snippet in snippetsList:
                if snippet not in mutatedSnippet:
                    mutatedSnippet.append(snippet)
                    tempMessage = seed.M[i].raw["Content"]

                    # ========  BitFlip ========
                    print("--BitFlip")
                    message = seed.M[i].raw["Content"]
                    asc = ""
                    for o in range(snippet[0], snippet[1]):
                        #print(255-ord(message[o]))
                        asc=asc+(chr(255-ord(message[o])))
                    #message[o] = chr(255-ord(chr(message[o])))
                    message = message[:snippet[0]] + asc + message[snippet[1] + 1:]
                    seed.M[i].raw["Content"] = message
                    responseHandle(seed, m.SnippetMutationSend(seed,i))
                    seed.M[i].raw["Content"] = tempMessage

                    # ========  Empty ========
                    print("--Empty")
                    message = seed.M[i].raw["Content"]
                    message = message[:snippet[0]] + message[snippet[1]+1:]
                    seed.M[i].raw["Content"] = message
                    responseHandle(seed, m.SnippetMutationSend(seed,i))
                    seed.M[i].raw["Content"] = tempMessage

                    # ========  Repeat ========
                    print("--Repeat")
                    message = seed.M[i].raw["Content"]
                    t = random.randint(2, 5)
                    message = message[:snippet[0]] + message[snippet[0]:snippet[1]] * t + message[snippet[1] + 1:]
                    seed.M[i].raw["Content"] = message
                    responseHandle(seed, m.SnippetMutationSend(seed,i))
                    seed.M[i].raw["Content"] = tempMessage

                    # ========  Interesting ========
                    print("--Interesting")
                    interestingString = ['on','off','True','False','0','1']
                    for t in interestingString:
                        message = seed.M[i].raw["Content"]
                        message = message[:snippet[0]] + t + message[snippet[1] + 1:]
                        seed.M[i].raw["Content"] = message
                        responseHandle(seed, m.SnippetMutationSend(seed,i))
                        seed.M[i].raw["Content"] = tempMessage

        seed.Snippet.append(mutatedSnippet)
    return 0


def Havoc(queue, restoreSeed):
    print("*Havoc")
    m = Messenger(restoreSeed)

    t = random.randint(0,len(queue)-1)
    seed = queue[t]

    i = random.randint(0,len(seed.M)-1)
    snippets = seed.Snippet[i]
    message = seed.M[i].raw["Content"]
    tempMessage = seed.M[i].raw["Content"]

    n = random.randint(0,len(snippets)-1)
    snippet = snippets[n]

    pick = random.randint(0,5)
    
    if pick == 0:  # ========  BitFlip ========
        asc = ""
        for o in range(snippet[0], snippet[1]):
            #print(255-ord(message[o]))
            asc=asc+(chr(255-ord(message[o])))
        #message[o] = chr(255-ord(chr(message[o])))
        message = message[:snippet[0]] + asc + message[snippet[1] + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp

    elif pick == 1: # ========  Empty ==========
        message = seed.M[i].raw["Content"]
        message = message[:snippet[0]] + message[snippet[1]+1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp
    
    elif pick == 2: # ========  Repeat ========
        message = seed.M[i].raw["Content"]
        t = random.randint(2, 5)
        message = message[:snippet[0]] + message[snippet[0]:snippet[1]] * t + message[snippet[1] + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp

    elif pick == 3: # ========  Interesting ========
        interestingString = ['on','off','True','False','0','1']
        interesting = random.randint(0,5)
        t = interestingString[interesting]
        message = seed.M[i].raw["Content"]
        message = message[:snippet[0]] + t + message[snippet[1] + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp
    
    elif pick == 4: # ======== Random Bytes Flip ===========
        start = random.randint(0,len(message)-1)
        end = random.randint(start,len(message))
        asc = ""
        for o in range(start, end):
            asc=asc+(chr(255-ord(message[o])))
        message = message[:start] + asc + message[end + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp

    return True



def getArgs(argv):
    inputfold = ''
    outputfold = ''
    restorefile = ''
    recordfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:r:o:c:", ["ifold=", "rfile=", "ofold=", "cfile="])
    except getopt.GetoptError:
        print('Snipuzz.py -i <inputfold> -r <restrefile> -o <outputfold> (-c <recordfile>)')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfold> -r <restrefile> -o <outputfold> (-c <recordfile>)')
            sys.exit()
        elif opt in ("-i", "--ifold"):
            inputfold = arg
        elif opt in ("-r", "--rfile"):
            restorefile = arg
        elif opt in ("-o", "--ofold"):
            outputfold = arg
        elif opt in ("-c", "--cfile"):
            recordfile = arg
        if not recordfile:
            recordfile = 'unavailable'
    print('Input fold：', inputfold)
    print('Restore file: ', restorefile)
    print('Output fold：', outputfold)
    print('Record file：', recordfile)

    return inputfold, restorefile, outputfold, recordfile


# python "e:/ondrive-file/OneDrive - Swinburne University/Desktop/Snipuzz-py/Snipuzz.py" -i "E:/ondrive-file/OneDrive - Swinburne University/Desktop/Snipuzz-py/in" -r "E:\ondrive-file\OneDrive - Swinburne University\Desktop\Snipuzz-py\record\yeelight-restore.txt" -o "E:\ondrive-file\OneDrive - Swinburne University\Desktop\Snipuzz-py\out" -c "E:\ondrive-file\OneDrive - Swinburne University\Desktop\Snipuzz-py\out\ProbeRecord.txt"

# python "e:/ondrive-file/OneDrive - Swinburne University/Desktop/Snipuzz-py/Snipuzz.py" -i "E:/ondrive-file/OneDrive - Swinburne University/Desktop/Snipuzz-py/in" -r "E:\ondrive-file\OneDrive - Swinburne University\Desktop\Snipuzz-py\record\yeelight-restore.txt" -o "E:\ondrive-file\OneDrive - Swinburne University\Desktop\Snipuzz-py\out" 

def main(argv):
    global queue, restoreSeed, outputfold

    inputfold, restorefile, outputfold, recordfile = getArgs(argv)
    if restorefile.endswith('.bin'):
        restoreSeed = readBinarySeed(restorefile)


    #queue =  readInputFold(inputfold)

    
    # If we are using a record file to initialize, use record file (which contains prior probe results)
    if recordfile and os.path.exists(recordfile):
        queue = readRecordFile(recordfile)
        for seed in queue:
            seed.display()
        if (dryRun(queue)):  # Dry Run
            print('#### Dry run failed, check the inputs or connection.')
            sys.exit()
    # Else we initialize via input folder and do initial probe
    else:
        queue = readBinInputFold(inputfold)
        if (dryRun(queue)):  # Dry Run
            print('#### Dry run failed, check the inputs or connection.')
            sys.exit()
        for i in range(len(queue)):
            # STATS
            start_time_probe = time.time()
            queue[i] = Probe(queue[i])
            elapsed_time_probe = time.time() - start_time_probe
        #writeRecord(queue, outputfold)
    

    skip = False
    while (1):
        if not skip:
            i=0
            while i < len(queue):
                if not queue[i].isMutated:
                    # STATS
                    start_time_snippetMutate = time.time()
                    SnippetMutate(queue[i], restoreSeed)
                    elapsed_time_snippetMutate = time.time() - start_time_snippetMutate
                i=i+1
        skip = True
        # STATS
        start_time_havoc = time.time()
        skip = Havoc(queue, restoreSeed)
        elapsed_time_havoc = time.time() - start_time_havoc


if __name__ == "__main__":
    main(sys.argv[1:])
