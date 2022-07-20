import re
import jieba

word_dict = set()

for i in range(1, 1001):
    file_path = 'E:\\OneDrive\\Coding\\Scraper Project\\TechNewsSpider\\TechNewsSpider\\TechData\\'+str(i) + '.txt'
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        article = f.read()
        article = article.replace("[","")
        article = article.replace("]","")
        article = article.replace("\\","")
        article = article.replace("n ","")


        punctuation = ", ：:＃＄％＆＇（）＊＋，－：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞–—‘’‛“”„‟…‧﹏ '"
        article = re.sub(r'[{}]+'.format(punctuation), '', article)
        after_process = ''
        pattern = r'[\r|。|！|!|?|？|｡]+'
        result_list = re.split(pattern, article)
        for sentence in result_list:
            sentence = re.sub('[^\u4e00-\u9fa5]', '', sentence)
            result = jieba.lcut(sentence)
            sentence_after_p = ''
            for word in result:
                if word.strip().lstrip() not in word_dict and word:
                    word_dict.add(word.strip().lstrip())
                    # sentence_after_p = sentence_after_p + word.strip().lstrip() + ' '
            sentence_after_p = " ".join(result)
            after_process = after_process + sentence_after_p + '\n'
            # result = " ".join(result)
            # print(result)

        
        cur_path2 = os.path.dirname(__file__)
        file_path2 = cur_path2 + '\\预处理结果\\' + str(i) + '.txt'
        with open(file_path2, 'w', encoding='utf-8') as f_save:
            f_save.write(after_process)

stop_words = []
with open(cur_path2+'\\停止词.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        stop_words.append(line)

word_dict_without_stop_word = []
for word in word_dict:
    if word not in stop_words:
        word_dict_without_stop_word.append(word)

final_word_dict = ''
n = 0
for word in word_dict.keys():
    n += 1
    final_word_dict = final_word_dict + str(n) + ' ' + word + ' ' + str(word_dict[word]) + '\n'


cur_path3 = os.path.dirname(__file__)
file_path3 = cur_path3 + '\\word_dict.txt'
with open(file_path3, 'w', encoding='utf-8') as f_save:
    f_save.write(final_word_dict)
