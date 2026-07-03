import json
import sys
import math
import io

def load_data():
    with open('./1_word.txt','r',encoding='utf-8')as f:
        one_word=json.load(f)
        #json格式的文件
    with open('./2_word.txt','r',encoding='utf-8')as f:
        two_word=json.load(f)
    word2pinyin={}
    with open('./word2pinyin.txt','r',encoding='utf-8')as f:
        for line in f:
            line =line.strip()
            if not line:
                continue
            parts=line.split()
            if len(parts)>=2:
                char=parts[0]
                pinyin=parts[1].lower()
                word2pinyin[char]=pinyin

    #拼音->汉字列表 词典   
    # 也就是拼音是key（比如da），words列表是value（比如） 
       #初始化一个空字典
    pinyin2words={}
    # p 是拼音（比如 "nen"），info 是对应的小字典（{"words":[...], "counts":[...]}）
    for p,info in one_word.items():
        #pinyin2words[p]=info['words']
        # 按频次从高到低排序候选字
        pairs = sorted(zip(info['counts'], info['words']), reverse=True)
        pinyin2words[p] = [w for _, w in pairs]

    # 用 word2pinyin 补充 pinyin2words，处理 1_word.txt 未覆盖的拼音
    # word2pinyin: {字: 拼音}，需要反转成 {拼音: [字]}
    for char, pinyin in word2pinyin.items():
        if pinyin not in pinyin2words:
            pinyin2words[pinyin] = []
        if char not in pinyin2words[pinyin]:
            pinyin2words[pinyin].append(char)
        # 同时确保 p1 里有该字的概率（生僻字可能不在 word_total_count 里）


                #pinyin2words的格式：
                #pinyin2words = {
                    #"nen": ["嫩", "恁"],
                    #"qing": ["清", "庆", "情"],
                    #"hua": ["华", "花", "划"]
                #}
    #一元概率：p1[w]=log(该字出现的总次数/所有字出现的总次数)
    #先统计每一个字的总count（包括所有读音的求和）
    #处理多音字
    word_total_count={}
    #word_total_count[清]=2000
    #grand_total是所有字的所有读音的求和
    grand_total=0
    for p, info in one_word.items():
        for w, c in zip(info['words'], info['counts']):
            word_total_count[w] = word_total_count.get(w, 0) + c
            grand_total+=c
    
    #构建一元概率词典(只考虑一个词出现的概率)
    #key是词，value是概率
    #总共有多少个字
    V = len(word_total_count)
    p1={}
    for w, c in word_total_count.items():
        p1[w] = math.log((c + 1) / (grand_total+V))
    # 补充 p1：word2pinyin 里有但 1_word.txt 没有的字，给一个极小概率
    for char in word2pinyin:
        if char not in p1:
            p1[char] = math.log(1 / (grand_total + V)) 

    #构建二元概率词典（考虑两个词出现的频率）
    #二元词典不是两个字同时出现的概率，而是出现前一个字的前提下，后一个字出现的概率
    #结构：p2["清"]["华"]=？
    p2={}
    #统计以每个w1开头的字对总计数
    w1_count={}
    for p,info in two_word.items():
        words=info['words']#words=["巴 嫩","把 嫩","拔 嫩"]
        counts=info['counts']
        for w_pair,c in zip(words,counts):
            #计算w1_count
            w1,_=w_pair.split()
            w1_count[w1] = w1_count.get(w1, 0) + c
    #计算p2
    for p,info in two_word.items():
        words=info['words']
        counts=info['counts']
        for w_pair,c in zip(words,counts):
            w1,w2=w_pair.split()
            if w1 not in p2:
                p2[w1]={}
            p2[w1][w2]=math.log((c+1)/(w1_count[w1] + V))
    return pinyin2words,p1,p2

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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    pinyin2words,p1,p2=load_data()
    for line in sys.stdin:#逐行读取输入
        line=line.strip()#去掉字符串首尾的空白
        if not line:#判断是否是空行
            continue
        pinyin_list=line.split()#以空格分割成pinyin_list
        chinese=viterbi(pinyin_list,pinyin2words,p1,p2)
        print(chinese)

if __name__=='__main__':
    main()      
