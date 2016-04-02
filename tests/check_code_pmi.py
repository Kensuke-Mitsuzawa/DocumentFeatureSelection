#coding:utf-8
import codecs
import math
import sys
from collections import defaultdict
# this code is from http://aidiary.hatenablog.com/entry/20100619/1276950312
# checked to work under python2.5

def mutual_information(target, data, k=5):
    # comment inputはlist 0th indexにカテゴリラベルがある。 1-th indexはすべてfeature word


    """カテゴリtargetにおける相互情報量が高い上位k件の単語を返す"""
    # 上位k件を指定しないときはすべて返す

    V = set()
    N11 = defaultdict(float)  # N11[word] -> wordを含むtargetの文書数
    N10 = defaultdict(float)  # N10[word] -> wordを含むtarget以外の文書数
    N01 = defaultdict(float)  # N01[word] -> wordを含まないtargetの文書数
    N00 = defaultdict(float)  # N00[word] -> wordを含まないtarget以外の文書数
    Np = 0.0  # targetの文書数
    Nn = 0.0  # target以外の文書す

    # N11とN10をカウント
    for d in data:
        cat, words = d[0], d[1:]
        if cat == target:
            Np += 1
            for wc in words:
                if ':' in wc: word, count = wc.split(":")
                else: word = wc

                V.add(word)
                N11[word] += 1  # 文書数をカウントするので+1すればOK
        elif cat != target:
            Nn += 1
            for wc in words:
                if ':' in wc: word, count = wc.split(":")
                else: word = wc

                V.add(word)
                N10[word] += 1

    # N01とN00は簡単に求められる
    for word in V:
        N01[word] = Np - N11[word]
        N00[word] = Nn - N10[word]

    for w, c in N01.items():
        if c < 0: N01[w] = 0.0

    for w, c in N00.items():
        if c < 0: N00[w] = 0.0

    # 総文書数
    N = Np + Nn


    # 各単語の相互情報量を計算
    MI = []
    for word in V:
        n11, n10, n01, n00 = N11[word], N10[word], N01[word], N00[word]
        # いずれかの出現頻度が0.0となる単語はlog2(0)となってしまうのでスコア0とする
        if n11 == 0.0 or n10 == 0.0 or n01 == 0.0 or n00 == 0.0:
            MI.append( (0.0, word) )
            continue
        # 相互情報量の定義の各項を計算
        temp1 = n11/N * math.log((N*n11)/((n10+n11)*(n01+n11)), 2)
        temp2 = n01/N * math.log((N*n01)/((n00+n01)*(n01+n11)), 2)
        temp3 = n10/N * math.log((N*n10)/((n10+n11)*(n00+n10)), 2)
        temp4 = n00/N * math.log((N*n00)/((n00+n01)*(n00+n10)), 2)
        score = temp1 + temp2 + temp3 + temp4
        MI.append( (score, word) )

    # 相互情報量の降順にソートして上位k個を返す
    MI.sort(reverse=True)
    return MI[0:k]


if __name__ == '__main__':

    input_data = [
        ['label_a', "I", "aa", "aa", "aa", "aa", "aa"],
        ['label_a', "bb", "aa", "aa", "aa", "aa", "aa"],
        ['label_a', "I", "aa", "hero", "some", "ok", "aa"],
        ['label_b', "bb", "bb", "bb"],
        ['label_b', "bb", "bb", "bb"],
        ['label_b', "hero", "ok", "bb"],
        ['label_b', "hero", "cc", "bb"],
        ['label_c', "cc", "cc", "cc"],
        ['label_c', "cc", "cc", "bb"],
        ['label_c', "xx", "xx", "cc"],
        ['label_c', "aa", "xx", "cc"],
    ]
    res = mutual_information(target='label_a', data=input_data, k=30)
    import pprint
    print('label_a')
    pprint.pprint(res)

    print('label_b')
    pprint.pprint(mutual_information(target='label_b', data=input_data, k=30))

    print('label_c')
    pprint.pprint(mutual_information(target='label_c', data=input_data, k=30))