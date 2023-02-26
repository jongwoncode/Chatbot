import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;)(])"
PAD = '<PAD>'   # 어떤 의미도 없는 패딩 토큰
STD = '<STD>'   # 시작 토큰을 의미
END = '<END>'   # 종료 토큰을 의미
UNK = '<UNK>'   # 사전에 없는 단어를 의미

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER =re.compile(FILTERS)
MAX_SEQUENCE =25

# data load
def load_data(path) :
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    return question, answer

# tokenizing 1 : filter and split
def data_tokenizer(data) :
    words = []
    for sentence in data :
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split() :
            words.append(word)
    return [word for word in words if word]

# tokenizing 2 : split with morphs
def prepro_like_morphlized(data) :
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data) :
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)
    return result_data

# make vcabulary
def load_vocabulary(path, vocab_path, tokenize_as_morph=False) :
    vocabulary_list = []
    if not os.path.exists(vocab_path) :
        if os.path.exists(path) :
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            if tokenize_as_morph :
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            
            data = []
            data.extend(question)
            data.extend(answer)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER
        
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file :
            for word in words :
                vocabulary_file.write(word + '\n')
    
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file :
        for line in vocabulary_file :
            vocabulary_list.append(line.strip())
    word2idx, idx2word = make_vocabulary(vocabulary_list)
    return word2idx, idx2word, len(word2idx)

# make : {word : index} dictionary, {index : word} dictionary
def make_vocabulary(vocabulary_list) :
    word2idx, idx2word = {}, {}
    for idx, word in enumerate(vocabulary_list) :
        word2idx[word] = idx    # {word : index} 
        idx2word[idx] = word    # {index : word} 
    return word2idx, idx2word



# encoder 전처리 함수
def enc_processing(value, dictionary, tokenize_as_morph=False) :
    sequences_input_index = []
    sequences_length = []

    if tokenize_as_morph : 
        value = prepro_like_morphlized(value)

    for sequence in value :
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        for word in sequence.split() :
            if dictionary.get(word) is not None :
                sequence_index.extend([dictionary[word]])
            else :
                sequence_index.extend([dictionary[UNK]])

        # 문장의 길이가 MAX_SEQUENCE 보다 길면. 끝 부분 제거
        if len(sequence_index) > MAX_SEQUENCE :
            sequence_index = sequence_index[:MAX_SEQUENCE]
        sequences_length.append(len(sequence_index))

        # 문장의 길이가 MAX_SEQUENCE 보다 짧거나 같다면. 차이만큼 <PAD> 추가
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    # return : 전처리한 데이터, padding 전 실제 길이
    return np.asarray(sequences_input_index), sequences_length


# decoder 전처리 함수 : 디코더의 입력으로 사용될 입력값을 전처리 하는 함수.
# "그래 오랜만이야" -> "<SOS>, 그래, 오랜만이야, <PAD>"
def dec_output_processing(value, dictionary, tokenize_as_morph=False) :
    sequences_output_index = []
    sequences_length = []

    if tokenize_as_morph :
        value = prepro_like_morphlized(value)
    
    for sequence in value :
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        # 시작 기호 <SOS>를 입력 
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        # 문장의 길이가 MAX_SEQUENCE 보다 길면. 끝 부분 제거
        if len(sequence_index) > MAX_SEQUENCE :
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequences_length.append(len(sequence_index))
        # 문장의 길이가 MAX_SEQUENCE 보다 짧거나 같다면. 차이만큼 <PAD> 추가
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)
            
    return np.asarray(sequences_output_index), sequences_length

# decoder 전처리 함수 : 디코더 결과로 학습을 위해 필요한 라벨인 타깃값을 만드는 전처리 함수.
# "그래 오랜만이야" -> "그래, 오랜만이야, <END>, <PAD>"
def dec_target_processing(value, dictionary, tokenize_as_morph=False) :
    sequences_targer_index = []

    if tokenize_as_morph :
        value = prepro_like_morphlized(value)
    
    for sequence in value :
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        #  종료 기호 <END> 입력 : 문장의 길이가 MAX_SEQUENCE 보다 길거나 같다면. 끝 부분 제거 & 종료 기호 입력
        if len(sequence_index) >= MAX_SEQUENCE :
            sequence_index = sequence_index[:MAX_SEQUENCE -1] + [dictionary[END]]
        else :
            sequence_index += [dictionary[END]]    
        # 문장의 길이가 MAX_SEQUENCE 보다 짧거나 같다면. 차이만큼 <PAD> 추가
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_targer_index.append(sequence_index)
            
    return np.asarray(sequences_targer_index)