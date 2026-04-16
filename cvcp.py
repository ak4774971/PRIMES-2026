import math
from collections import defaultdict

def getProbabilities(corpusTokens,windowSize):
    singleCounts = defaultdict(int)
    pairCounts = defaultdict(int)

    numWindows = max(0,len(corpusTokens) - windowSize + 1)
    if numWindows == 0:
        return {},{}

    for i in range(numWindows):
        window = set(corpusTokens[i : i + windowSize])
        for w in window:
            singleCounts[w] += 1

        window_list = list(window)
        for j in range(len(window_list)):
            for k in range(j + 1,len(window_list)):
                w1,w2 = sorted((window_list[j],window_list[k]))
                pairCounts[(w1,w2)] += 1

    singleProbs = {w: c / numWindows for w,c in singleCounts.items()}
    pairProbs = {p: c / numWindows for p,c in pairCounts.items()}
    return singleProbs,pairProbs

def compute_cp(topic,corpusTokens,windowSize=5):
    singleProbs,pairProbs = getProbabilities(corpusTokens,windowSize)
    scores = []

    for i in range(1,len(topic)):
        for j in range(i):
            wi,wj = topic[i],topic[j]
            p_i = singleProbs.get(wi,0.0)
            p_j = singleProbs.get(wj,0.0)
            p_ij = pairProbs.get(tuple(sorted((wi,wj))),0.0)

            p_i_given_j = p_ij / p_j if p_j > 0 else 0.0
            p_i_given_not_j = (p_i - p_ij) / (1.0 - p_j) if p_j < 1.0 else 0.0

            num = p_i_given_j - p_i_given_not_j
            den = p_i_given_j + p_i_given_not_j
            mf = num / den if den > 0 else 0.0
            scores.append(mf)

    return sum(scores) / len(scores) if scores else 0.0

def cosineSim(v1,v2):
    dot = sum(x * y for x,y in zip(v1,v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0

def computeCv(topic,corpusTokens,windowSize=10):
    singleProbs,pairProbs = getProbabilities(corpusTokens,windowSize)
    vectors = []

    for wi in topic:
        vec = []
        for wk in topic:
            p_i = singleProbs.get(wi,0.0)
            p_k = singleProbs.get(wk,0.0)
            p_ik = pairProbs.get(tuple(sorted((wi,wk))),0.0)

            if p_ik == 0.0 or p_i == 0.0 or p_k == 0.0:
                npmi = 0.0
            else:
                pmi = math.log(p_ik / (p_i * p_k))
                npmi = pmi / -math.log(p_ik)
            vec.append(npmi)
        vectors.append(vec)

    N = len(topic)
    centroid = [sum(vectors[j][k] for j in range(N)) for k in range(N)]

    scores = [cosineSim(vectors[i],centroid) for i in range(N)]
    return sum(scores) / len(scores) if scores else 0.0

corpus = "The quick brown fox jumps over the lazy dog. A dog is a man's best friend."
corpusTokens = corpus.lower().replace('.','').split()
topicWords = ["dog","fox","quick","brown"]

cp_score = compute_cp(topicWords,corpusTokens)
cv_score = computeCv(topicWords,corpusTokens)

print(cp_score,cv_score)


