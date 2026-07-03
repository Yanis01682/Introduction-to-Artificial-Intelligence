# 拼音输入法二元模型

## 一、程序执行指令
支持通过标准输入输出重定向运行拼音转换程序。在cmd终端执行以下指令完成数据读取与预测输出。
在项目根目录下：
python main.py < data/input.txt > data/output.txt
python src/evaluate.py
python src/main_trigram.py < data/input.txt > data/output_trigram.txt
python src/evaluate_trigram.py

## 二、项目目录结构约束
项目严格遵守测评要求的文件层级组织方式。压缩包内不包含任何中间文件与语料库文件。
1. corpus目录放置sina_news_gbk语料库
2. data目录放置标准答案与测试文件及词表
3. src目录放置辅助源代码与评测脚本
4. main.py主程序入口位于项目根目录
5. readme.md说明文档位于项目根目录
6. requirements.txt依赖文件位于项目根目录

## 三、代码底层逻辑说明
1. main函数程序启动后执行 build_pinyin_table 函数解析数据表构建合法汉字集合与映射字典。
2. 执行 train_model 函数扫描 corpus 目录读取 gbk 编码文件。
3. 剔除无效字符后将文本按空格切分。
4. 遍历提取一元字频与二元相邻字频并计算对数转移概率。
5. 拦截标准输入流逐行截取拼音序列输入 viterbi 函数。
6. 动态规划计算所有潜在汉字序列概率路径。
7. 反向回溯取得最大概率路径后拼接为字符串格式写入标准输出流。