def evaluate(out_path, ans_path):
    with open(out_path, 'r', encoding='utf-8') as f_out, \
         open(ans_path, 'r', encoding='utf-8') as f_ans:
        
        out_lines = f_out.readlines()
        ans_lines = f_ans.readlines()
        
        total_sentences = len(ans_lines)
        correct_sentences = 0
        total_chars = 0
        correct_chars = 0
        
        for out_line, ans_line in zip(out_lines, ans_lines):
            out_line = out_line.strip()
            ans_line = ans_line.strip()
            
            # 统计句准率
            if out_line == ans_line:
                correct_sentences += 1
            
            # 统计字准率
            total_chars += len(ans_line)
            # 逐字比对，取两者长度的最小值防止溢出
            for i in range(min(len(out_line), len(ans_line))):
                if out_line[i] == ans_line[i]:
                    correct_chars += 1
                    
        print(f"句子准确率: {correct_sentences / total_sentences * 100:.2f}%")
        print(f"汉字准确率: {correct_chars / total_chars * 100:.2f}%")

if __name__ == '__main__':
    evaluate('./data/output.txt', './data/answer.txt')