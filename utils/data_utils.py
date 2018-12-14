import re
import jieba
import codecs
import collections   # 统计词频
import numpy as np

# sentence polarity dataset v1.0 from http://www.cs.cornell.edu/people/pabo/movie-review-data/


PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def get_tokens(sent):
    # Processing tokens
    _PAD = b"_PAD"
    _GO = b"_GO"
    _EOS = b"_EOS"
    _UNK = b"_UNK"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

    # Regular expressions used to tokenize.
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
    _DIGIT_RE = re.compile(br"\d")
    words = []
    for space_separated_fragment in sent.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def build_vocab(corpus):
    vocab = []
    vocab_freqs = collections.Counter()
    for item in corpus:
        clean_item = item.strip().split(" ")
        vocab.extend([word for word in clean_item if word])
    for word in vocab:
        vocab_freqs[word] += 1
    return vocab,vocab_freqs


def seg_corpus(corpus):
    seg_corpus = []
    for line in corpus:
        line = str(line).strip()
        seg_list = jieba.cut(line, cut_all=False)
        # 过滤空字符
        seg_list = [w for w in seg_list if w != ' ']
        seg_corpus.append(" ".join(seg_list))
    return seg_corpus


def load_data_and_labels():
    pass
    # # Load the data
    # positive_examples = list(open("data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # positive_examples = [get_tokens(clean_str(sent)) for sent in positive_examples]
    # negative_examples = list(open("data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # negative_examples = [get_tokens(clean_str(sent)) for sent in negative_examples]
    # X = positive_examples + negative_examples

    # # Labels
    # positive_labels = [[0,1] for _ in positive_examples]
    # negative_labels = [[1,0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)

    # print("Total: %i, NEG: %i, POS: %i" % (len(y), np.sum(y[:, 0]), np.sum(y[:, 1])))

    # return X, y

def create_vocabulary(X, max_vocabulary_size=5000):
    pass
    # vocab = {}
    # for sentence in X:
    #     for word in sentence:
    #         if word in vocab:
    #             vocab[word] += 1
    #         else:
    #             vocab[word] = 1

    # # Get list of all vocab words starting with [_PAD, _GO, _EOS, _UNK]
    # # and then words sorted by count
    # vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    # vocab_list = vocab_list[:max_vocabulary_size]

    # vocab_dict = dict((x,y) for (y,x) in enumerate(vocab_list))
    # rev_vocab_dict = {v: k for k, v in vocab_dict.items()}

    # print("Total of %i unique tokens" % len(vocab_list))
    # return vocab_list, vocab_dict, rev_vocab_dict


def split_dataset(X, y, seq_lens, train_ratio=0.8):
    pass
    # X = np.array(X)
    # seq_lens = np.array(seq_lens)
    # data_size = len(X)

    # # Shuffle the data
    # shuffle_indices = np.random.permutation(np.arange(data_size))
    # X, y, seq_lens = X[shuffle_indices], y[shuffle_indices], \
    #                  seq_lens[shuffle_indices]

    # # Split into train and validation set
    # train_end_index = int(train_ratio*data_size)
    # train_X = X[:train_end_index]
    # train_y = y[:train_end_index]
    # train_seq_lens = seq_lens[:train_end_index]
    # valid_X = X[train_end_index:]
    # valid_y = y[train_end_index:]
    # valid_seq_lens = seq_lens[train_end_index:]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # 移除网址
    string = re.sub(r"[a-zA-z]+://[^\s]*", " ", string)

    # 中文标点符号
    string = re.sub(r"\？", " ", string)
    string = re.sub(r"，", " ", string)
    string = re.sub(r"！", " ", string)
    string = re.sub(r"\（", " ", string)
    string = re.sub(r"\）", " ", string)
    string = re.sub(r"\《", " ", string)
    string = re.sub(r"\》", " ", string)
    string = re.sub(r"\、", " ", string)
    string = re.sub(r"\»", " ", string)
    string = re.sub(r"\。", " ", string)
    string = re.sub(r"\「", " ", string)
    string = re.sub(r"\」", " ", string)
    string = re.sub(r"\-", " ", string)
    string = re.sub(r"\【", " ", string)
    string = re.sub(r"\】", " ", string)
    string = re.sub(r"\：", " ", string)
    string = re.sub(r"\“", " ", string)
    string = re.sub(r"\”", " ", string)
    string = re.sub(r"\；", " ", string)

    string = re.sub(r"〔", " ", string)
    string = re.sub(r"〕", " ", string)
    string = re.sub(r"……", " ", string)
    string = re.sub(r"\¥", " ", string)
    string = re.sub(r"\@", " ", string)

    # 英文标点符号
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)

    string = re.sub(r"≈", " ", string)
    string = re.sub(r":", " ", string)
    string = re.sub(r"\~", " ", string)
    string = re.sub(r"\—", " ", string)
    string = re.sub(r"⋯", " ", string)
    string = re.sub(r"[p]\.[s]\.", " ", string)

    string = re.sub(r"\/"," ", string)
    string = re.sub(r"\?"," ", string)
    string = re.sub(r"="," ", string)
    string = re.sub(r"…"," ", string)

    string = re.sub(r"〜", " ", string)
    string = re.sub(r"～", " ", string)
    string = re.sub(r"–", "", string)
    string.replace(' ','')
    
    return string.strip().lower()


#######################################################
def create_word_dict(word_list):
    """
    Create a dictionary of words from a list of list of words.
    """
    assert type(word_list) is list
    word_dict = {}
    for words in word_list:
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    return word_dict

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    
    # 当python要做编码转换的时候，会借助于内部的编码，转换过程是这样的：原有编码 -> 内部编码 -> 目的编码， 
    # python的内部是使用unicode来处理的，但是unicode的使用需要考虑的是它的编码格式有两种，一是UCS-2，它一共有65536个码位，另一种是UCS-4，它有2147483648g个码位。
    # codecs专门用作编码转换，转换为 utf-8
    # for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
    #     line = line.rstrip().split()
    #     if len(line) == word_dim + 1:
    #         pre_trained[line[0]] = np.array(
    #             [float(x) for x in line[1:]]
    #         ).astype(np.float32)
    #     else:
    #         emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


stop_words = [
            "一","一下","一些","一切","一则","一天","一定","一方面","一旦","一时","一来",
            "一样","一次","一片","一直","一致","一般","一起","一边","一面","万一","上下",
            "上升","上去","上来","上述","上面","下列","下去","下来","下面","不一","不久",
            "不仅","不会","不但","不光","不单","不变","不只","不可","不同","不够","不如",
            "不得","不怕","不惟","不成","不拘","不敢","不断","不是","不比","不然","不特",
            "不独","不管","不能","不要","不论","不足","不过","不问","与","与其","与否",
            "与此同时","专门","且","两者","严格","严重","个","个人","个别","中小","中间",
            "丰富","临","为","为主","为了","为什么","为什麽","为何","为着","主张","主要",
            "举行","乃","乃至","么","之","之一","之前","之后","之後","之所以","之类",
            "乌乎","乎","乘","也","也好","也是","也罢","了","了解","争取","于","于是",
            "于是乎","云云","互相","产生","人们","人家","什么","什么样","什麽","今后","今天",
            "今年","今後","仍然","从","从事","从而","他","他人","他们","他的","代替",
            "以","以上","以下","以为","以便","以免","以前","以及","以后","以外","以後",
            "以来","以至","以至于","以致","们","任","任何","任凭","任务","企图","伟大",
            "似乎","似的","但","但是","何","何况","何处","何时","作为","你","你们",
            "你的","使得","使用","例如","依","依照","依靠","促进","保持","俺","俺们",
            "倘","倘使","倘或","倘然","倘若","假使","假如","假若","做到","像","允许",
            "充分","先后","先後","先生","全部","全面","兮","共同","关于","其","其一",
            "其中","其二","其他","其余","其它","其实","其次","具体","具体地说","具体说来",
            "具有","再者","再说","冒","冲","决定","况且","准备","几","几乎","几时","凭",
            "凭借","出去","出来","出现","分别","则","别","别的","别说","到","前后",
            "前者","前进","前面","加之","加以","加入","加强","十分","即","即令","即使",
            "即便","即或","即若","却不","原来","又","及","及其","及时","及至","双方",
            "反之","反应","反映","反过来","反过来说","取得","受到","变成","另","另一方面",
            "另外","只是","只有","只要","只限","叫","叫做","召开","叮咚","可","可以",
            "可是","可能","可见","各","各个","各人","各位","各地","各种","各级","各自",
            "合理","同","同一","同时","同样","后来","后面","向","向着","吓","吗","否则",
            "吧","吧哒","吱","呀","呃","呕","呗","呜","呜呼","呢","周围","呵","呸",
            "呼哧","咋","和","咚","咦","咱","咱们","咳","哇","哈","哈哈","哉","哎",
            "哎呀","哎哟","哗","哟","哦","哩","哪","哪个","哪些","哪儿","哪天","哪年",
            "哪怕","哪样","哪边","哪里","哼","哼唷","唉","啊","啐","啥","啦","啪达",
            "喂","喏","喔唷","嗡嗡","嗬","嗯","嗳","嘎","嘎登","嘘","嘛","嘻","嘿",
            "因","因为","因此","因而","固然","在","在下","地","坚决","坚持","基本",
            "处理","复杂","多","多少","多数","多次","大力","大多数","大大","大家","大批",
            "大约","大量","失去","她","她们","她的","好的","好象","如","如上所述","如下",
            "如何","如其","如果","如此","如若","存在","宁","宁可","宁愿","宁肯","它",
            "它们","它们的","它的","安全","完全","完成","实现","实际","宣布","容易","密切",
            "对","对于","对应","将","少数","尔后","尚且","尤其","就","就是","就是说",
            "尽","尽管","属于","岂但","左右","巨大","巩固","己","已经","帮助","常常",
            "并","并不","并不是","并且","并没有","广大","广泛","应当","应用","应该","开外",
            "开始","开展","引起","强烈","强调","归","当","当前","当时","当然","当着",
            "形成","彻底","彼","彼此","往","往往","待","後来","後面","得","得出","得到",
            "心里","必然","必要","必须","怎","怎么","怎么办","怎么样","怎样","怎麽","总之",
            "总是","总的来看","总的来说","总的说来","总结","总而言之","恰恰相反","您","意思",
            "愿意","慢说","成为","我","我们","我的","或","或是","或者","战斗","所",
            "所以","所有","所谓","打","扩大","把","抑或","拿","按","按照","换句话说",
            "换言之","据","掌握","接着","接著","故","故此","整个","方便","方面","旁人",
            "无宁","无法","无论","既","既是","既然","时候","明显","明确", "是","是否",
            "是的","显然","显著","普通","普遍","更加","曾经","替","最后","最大","最好",
            "最後","最近","最高","有","有些","有关","有利","有力","有所","有效","有时",
            "有点","有的","有着","有著","望","朝","朝着","本","本着","来","来着","极了",
            "构成","果然","果真","某","某个","某些","根据","根本","欢迎","正在","正如",
            "正常","此","此外","此时","此间","毋宁","每","每个","每天","每年","每当",
            "比","比如","比方","比较","毫不","没有","沿","沿着","注意","深入","清楚",
            "满足","漫说","焉","然则","然后","然後","然而","照","照着","特别是","特殊",
            "特点","现代","现在","甚么","甚而","甚至","用","由","由于","由此可见","的",
            "的话","目前","直到","直接","相似","相信","相反","相同","相对","相对而言","相应",
            "相当","相等","省得","看出","看到","看来","看看","看见","真是","真正","着",
            "着呢","矣","知道","确定","离","积极","移动","突出","突然","立即","第","等",
            "等等","管","紧接着","纵","纵令","纵使","纵然","练习","组成","经","经常",
            "经过","结合","结果","给","绝对","继续","继而","维持","综上所述","罢了","考虑",
            "者","而","而且","而况","而外","而已","而是","而言","联系","能","能否",
            "能够","腾","自","自个儿","自从","自各儿","自家","自己","自身","至","至于",
            "良好","若","若是","若非","范围","莫若","获得","虽","虽则","虽然","虽说",
            "行为","行动","表明","表示","被","要","要不","要不是","要不然","要么","要是",
            "要求","规定","觉得","认为","认真","认识","让","许多","论","设使","设若",
            "该","说明","诸位","谁","谁知","赶","起","起来","起见","趁","趁着","越是",
            "跟","转动","转变","转贴","较","较之","边","达到","迅速","过","过去","过来",
            "运用","还是","还有","这","这个","这么","这么些","这么样","这么点儿","这些",
            "这会儿","这儿","这就是说","这时","这样","这点","这种","这边","这里","这麽",
            "进入","进步","进而","进行","连","连同","适应","适当","适用","逐步","逐渐",
            "通常","通过","造成","遇到","遭到","避免","那","那个","那么","那么些","那么样",
            "那些","那会儿","那儿","那时","那样","那边","那里","那麽","部分","鄙人","采取",
            "里面","重大","重新","重要","鉴于","问题","防止","阿","附近","限制","除",
            "除了","除此之外","除非","随","随着","随著","集中","需要","非但","非常","非徒",
            "靠","顺","顺着","首先","高兴","是不是","说说","，","。","《","》","？","『","！",",",".","!","?","、","\"","\"",
            "月","日","年","“","…","”","】","【","（","）"
        ]