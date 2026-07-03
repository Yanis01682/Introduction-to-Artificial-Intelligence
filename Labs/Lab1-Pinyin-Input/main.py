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
def train_model(corpus_dir,valid_chars):
    word_total_count={c:0 for c in valid_chars}
    #word_total_count[清]=2，存每个字出现的总次数
    pair_count={}
    #存字对的出现次数，如清后面接华出现了多少次

    #扫描文件夹，只处理后缀是.txt的文件，拼接成完整的文件路径
    for filename in os.listdir(corpus_dir):
        if not filename.endswith('.txt') or filename=='README.txt':
            continue
        filepath=os.path.join(corpus_dir,filename)
        
        with open(filepath,'r',encoding='gbk',errors='ignore')as f:
            for line in f:
                #遍历这一行的每一个字符，如果该字符在6763个字表中就保留，不在的话就换成空格
                #适用于json格式
                clean_line="".join([c if c in valid_chars else " " for c in line])
                # 来吧冠军 李娜姜山首秀 劲爆亮相秀球技
                #split切分，结果是一个纯汉字片段组成的列表 ["来吧冠军"，"李娜姜山首秀"，"劲爆亮相秀球技"]
                sentences = clean_line.split()

                for seq in sentences:
                    for i in range(len(seq)):
                        c1=seq[i]
                        word_total_count[c1]+=1
                        if i<len(seq)-1:
                            c2=seq[i+1]
                            if c1 not in pair_count:
                                pair_count[c1] = {}
                            pair_count[c1][c2] = pair_count[c1].get(c2, 0) + 1
        
    V=len(valid_chars)
    grand_total=sum(word_total_count.values())
    #p1,p2构建一元概率和二元概率
    p1={}
    for c in valid_chars:
        p1[c]=math.log((word_total_count[c]+1)/(grand_total+V))
    p2={}
    w1_count={}
    for w1,following in pair_count.items():
        w1_count[w1] = sum(following.values())
    for w1, following in pair_count.items():
        p2[w1] = {}
        for w2, count_12 in following.items():
            p2[w1][w2] = math.log((count_12 + 1) / (w1_count[w1] + V))
            
    return p1, p2

def viterbi(pinyin_list,pinyin2words,p1,p2):
    n=len(pinyin_list)
    if n==0:
        return ""
    dp=[{}for _ in range(n)]#[{}, {}, {}, {}]
    #后续赋值dp[0] = {"清": -0.01, "庆": -4.5, "情": -2.1}
    path=[{}for _ in range(n)]

    #处理第一个拼音
    first_pin=pinyin_list[0]
    if first_pin not in pinyin2words:
        return ""
    first_words = pinyin2words[first_pin]
    for w in first_words:
        dp[0][w] = p1.get(w, -20.0)  # 用 .get 避免 KeyError
        path[0][w]=None

    #处理第i个拼音
    for i in range(1,n):
        cur_pin=pinyin_list[i]#当前拼音
        pre_pin=pinyin_list[i-1]#前一个拼音
        if cur_pin not in pinyin2words or pre_pin not in pinyin2words:
            return ""
        cur_words = pinyin2words[cur_pin]
        pre_words = pinyin2words[pre_pin]

        for w2 in cur_words:
            max_prob=-float('inf')
            best_prev=None
            pre_dp = dp[i-1]
            for w1 in pre_words:
                LAMBDA = 0.95
                PENALTY = 0.1  # 没有二元证据时额外惩罚，可以试 0.5~0.9
                p1_w2 = p1.get(w2, -20.0)
                if w1 in p2 and w2 in p2[w1]:
                    mixed = LAMBDA * math.exp(p2[w1][w2]) + (1 - LAMBDA) * math.exp(p1_w2)
                    trans_prob = math.log(mixed) if mixed > 0 else p1_w2
                else:
                    trans_prob = p1_w2 + math.log(PENALTY)  # 加惩罚
                current_prob = pre_dp[w1] + trans_prob
                if current_prob > max_prob:
                    max_prob = current_prob
                    best_prev = w1
            if best_prev is not None:
                dp[i][w2] = max_prob
                path[i][w2] = best_prev
            else:
                dp[i][w2] = p1[w2]
                path[i][w2] = pre_words[0]

    #最后一个拼音处理完了，取dp最大的那个回溯
    last_pin=pinyin_list[n-1]
    last_words=pinyin2words[last_pin]
    w_last=max(last_words,key=lambda w:dp[n-1][w])

    result=[w_last]
    current_w=w_last
    for i in range(n-1,0,-1):#-1指定步长是-1
        if current_w not in path[i] or path[i][current_w] is None:
            pre_words = pinyin2words[pinyin_list[i-1]]
            current_w = max(pre_words, key=lambda w: dp[i-1][w])
        else:
            current_w = path[i][current_w]
        result.append(current_w)
    result.reverse()
    return ''.join(result)

def main():
    sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    table_path='./data/拼音汉字表.txt'
    corpus_dir='./corpus/sina_news_gbk'

    pinyin2words,valid_chars=build_pinyin_table(table_path)
    p1,p2=train_model(corpus_dir,valid_chars)
  
    for line in sys.stdin:
        line=line.strip()
        if not line:
            print("")
            continue
    #pinyin_list是输入序列
        pinyin_list=line.split()
        result=viterbi(pinyin_list,pinyin2words,p1,p2)
        print(result)

if __name__=='__main__':
    main()