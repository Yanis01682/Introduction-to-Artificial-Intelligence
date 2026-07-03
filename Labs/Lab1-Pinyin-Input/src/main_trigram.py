import sys
import math
import os
import io

#读取拼音汉字表，建议拼音到汉字的映射
def build_pinyin_table(filepath):

    #建立pinyin2words
    #顺便提取出一个valid_chars集合
    pinyin2words={}
    valid_chars=set()

    with open(filepath,'r',encoding='gbk')as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split()
            if len(parts)>=2:
                pinyin=parts[0].lower()
                chars=parts[1:]
                
                pinyin2words[pinyin]=chars
                valid_chars.update(chars)

    return pinyin2words,valid_chars

#从原始的新浪新闻文本中，通过统计汉字出现频率，生成 Viterbi 算法所需的概率表 p1（字概率）和 p2（转移概率）。
def train_model(corpus_dir, valid_chars):
    word_total_count = {c: 0 for c in valid_chars}
    pair_count = {}
    triple_count = {}

    for filename in os.listdir(corpus_dir):
        if not filename.endswith('.txt') or filename == 'README.txt':
            continue
        
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                clean_line = "".join([c if c in valid_chars else " " for c in line])
                sentences = clean_line.split()

                for seq in sentences:
                    for i in range(len(seq)):
                        c1 = seq[i]
                        word_total_count[c1] += 1
                        
                        if i < len(seq) - 1:
                            c2 = seq[i+1]
                            if c1 not in pair_count:
                                pair_count[c1] = {}
                            pair_count[c1][c2] = pair_count[c1].get(c2, 0) + 1
                            
                            if i < len(seq) - 2:
                                c3 = seq[i+2]
                                if c1 not in triple_count:
                                    triple_count[c1] = {}
                                if c2 not in triple_count[c1]:
                                    triple_count[c1][c2] = {}
                                triple_count[c1][c2][c3] = triple_count[c1][c2].get(c3, 0) + 1

    V = len(valid_chars)
    grand_total = sum(word_total_count.values())
    
    p1 = {}
    for c in valid_chars:
        p1[c] = math.log((word_total_count[c] + 1) / (grand_total + V))
        
    p2 = {}
    w1_count = {}
    for w1, following in pair_count.items():
        w1_count[w1] = sum(following.values())
    for w1, following in pair_count.items():
        p2[w1] = {}
        for w2, count_12 in following.items():
            p2[w1][w2] = math.log((count_12 + 1) / (w1_count[w1] + V))

    p3 = {}
    for w1, dict_w2 in triple_count.items():
        p3[w1] = {}
        for w2, dict_w3 in dict_w2.items():
            p3[w1][w2] = {}
            count_12 = pair_count[w1][w2]
            for w3, count_123 in dict_w3.items():
                p3[w1][w2][w3] = math.log((count_123 + 1) / (count_12 + V))

    return p1, p2, p3

def viterbi_trigram(pinyin_list, pinyin2words, p1, p2, p3):
    n = len(pinyin_list)
    if n == 0:
        return ""
    
    dp = [{} for _ in range(n)]
    path = [{} for _ in range(n)]

    first_pin = pinyin_list[0]
    if first_pin not in pinyin2words:
        return ""
    
    first_words = pinyin2words[first_pin]
    for w0 in first_words:
        if None not in dp[0]:
            dp[0][None] = {}
            path[0][None] = {}
        dp[0][None][w0] = p1.get(w0, -20.0)
        path[0][None][w0] = None

    if n == 1:
        w_last = max(first_words, key=lambda w: dp[0][None][w])
        return w_last

    second_pin = pinyin_list[1]
    if second_pin not in pinyin2words:
        return ""
    second_words = pinyin2words[second_pin]
    
    for w0 in first_words:
        for w1 in second_words:
            L2 = 0.95
            L1 = 0.05
            prob2 = math.exp(p2[w0][w1]) if w0 in p2 and w1 in p2[w0] else 0.0
            prob1 = math.exp(p1.get(w1, -20.0))
            mixed = L2 * prob2 + L1 * prob1
            trans_prob = math.log(mixed) if mixed > 0 else p1.get(w1, -20.0) - 2.3
            
            if w0 not in dp[1]:
                dp[1][w0] = {}
                path[1][w0] = {}
            dp[1][w0][w1] = dp[0][None][w0] + trans_prob
            path[1][w0][w1] = None

    for i in range(2, n):
        cur_pin = pinyin_list[i]
        if cur_pin not in pinyin2words:
            return ""
        cur_words = pinyin2words[cur_pin]
        pre_words = pinyin2words[pinyin_list[i-1]]
        pre_pre_words = pinyin2words[pinyin_list[i-2]]

        for w1 in pre_words:
            for w2 in cur_words:
                max_prob = -float('inf')
                best_w0 = None
                for w0 in pre_pre_words:
                    if w0 not in dp[i-1] or w1 not in dp[i-1][w0]:
                        continue
                    
                    L3 = 0.8
                    L2 = 0.15
                    L1 = 0.05
                    prob3 = math.exp(p3[w0][w1][w2]) if w0 in p3 and w1 in p3[w0] and w2 in p3[w0][w1] else 0.0
                    prob2 = math.exp(p2[w1][w2]) if w1 in p2 and w2 in p2[w1] else 0.0
                    prob1 = math.exp(p1.get(w2, -20.0))
                    mixed = L3 * prob3 + L2 * prob2 + L1 * prob1
                    trans_prob = math.log(mixed) if mixed > 0 else p1.get(w2, -20.0) - 2.3
                    
                    current_prob = dp[i-1][w0][w1] + trans_prob
                    if current_prob > max_prob:
                        max_prob = current_prob
                        best_w0 = w0
                
                if best_w0 is not None:
                    if w1 not in dp[i]:
                        dp[i][w1] = {}
                        path[i][w1] = {}
                    dp[i][w1][w2] = max_prob
                    path[i][w1][w2] = best_w0

    max_final_prob = -float('inf')
    best_last_w1 = None
    best_last_w2 = None
    
    last_words = pinyin2words[pinyin_list[n-1]]
    pre_last_words = pinyin2words[pinyin_list[n-2]]
    
    for w1 in pre_last_words:
        if w1 not in dp[n-1]:
            continue
        for w2 in last_words:
            if w2 in dp[n-1][w1] and dp[n-1][w1][w2] > max_final_prob:
                max_final_prob = dp[n-1][w1][w2]
                best_last_w1 = w1
                best_last_w2 = w2

    if best_last_w1 is None or best_last_w2 is None:
        return ""

    result = [best_last_w2, best_last_w1]
    curr_w1 = best_last_w1
    curr_w2 = best_last_w2
    
    for i in range(n-1, 1, -1):
        w0 = path[i][curr_w1][curr_w2]
        result.append(w0)
        curr_w2 = curr_w1
        curr_w1 = w0
        
    result.reverse()
    return ''.join(result)

def main():
    sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    table_path='./data/拼音汉字表.txt'
    corpus_dir='./corpus/sina_news_gbk'

    pinyin2words,valid_chars=build_pinyin_table(table_path)
    p1, p2, p3 = train_model(corpus_dir, valid_chars)

    for line in sys.stdin:
        line=line.strip()
        if not line:
            print("")
            continue
    #pinyin_list是输入序列
        pinyin_list=line.split()
        result=viterbi_trigram(pinyin_list,pinyin2words,p1,p2,p3)
        print(result)

if __name__=='__main__':
    main()