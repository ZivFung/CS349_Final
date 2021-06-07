import numpy as np
import ujson as json
import spacy
from tqdm import tqdm
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def process_data(data, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    for i in range(data.shape[0]):
        tweet=data["tidy_tweet"].iloc[i]
        tweet_tokens = word_tokenize(tweet)
        tweet_chars = [list(token) for token in tweet_tokens]
        for token in tweet_tokens:
            word_counter[token] += 1
            for char in token:
                char_counter[char] += 1
        example = {"context_tokens": tweet_tokens,
                   "context_chars": tweet_chars}
        examples.append(example)
    return examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=1193514):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def build_features( examples, data_type,  word2idx_dict, char2idx_dict,max_seq_len,max_char_len):

    print(f"Converting {data_type} examples to indices...")
    total=0
    tweet_idxs = []
    tweet_char_idxs = []

    for n, example in tqdm(enumerate(examples)):
        total += 1
        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        tweet_idx = np.zeros([max_seq_len], dtype=np.int32)
        tweet_char_idx = np.zeros([max_seq_len, max_char_len], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            if i == max_seq_len:
                break
            tweet_idx[i] = _get_word(token)
        tweet_idxs.append(tweet_idx)

        for j, token in enumerate(example["context_chars"]):
            if j ==max_seq_len:
                break

            for k, char in enumerate(token):
                if k == max_char_len:
                    break
                tweet_char_idx[j, k] = _get_char(char)
        tweet_char_idxs.append(tweet_char_idx)

    print(f"Build totally {total} features from {data_type} file")

    return tweet_idxs,tweet_char_idxs


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)
def process_data(train_file_path,test_file_path,glove_file_path,save_dir):
    MAX_SEQ_LENGTH=20
    MAX_CHAR_LENGTH=10
    MAX_NUM_WORD=100000

    trainData = pd.read_csv(train_file_path)
    testData = pd.read_csv(test_file_path)

    dataSet = trainData.append(testData, ignore_index=True)

    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)

        return input_txt

    # remove twitter handles (@user)
    dataSet['tidy_tweet'] = np.vectorize(remove_pattern)(dataSet['tweet'], "@[\w]*")

    # remove special characters, numbers, punctuations
    dataSet['tidy_tweet'] = dataSet['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
    dataSet['tidy_tweet'] = dataSet['tidy_tweet'].str.replace("#", " ")

    # tokenize
    tokenized_Data = dataSet['tidy_tweet'].apply(lambda x: x.split())

    for i in range(len(tokenized_Data)):
        tokenized_Data[i] = ' '.join(tokenized_Data[i])
    dataSet['tidy_tweet'] = tokenized_Data

    trainData=dataSet[:trainData.shape[0]]
    testData=dataSet[trainData.shape[0]:]

    #process word data
    word_counter,char_counter=Counter(),Counter()
    train_examples=process_data(trainData,"train",word_counter,char_counter)


    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=glove_file_path, vec_size=100)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char',  vec_size=100)

    tweet_idxs,tweet_char_idxs=build_features(train_examples,"train",word2idx_dict,char2idx_dict,MAX_SEQ_LENGTH,MAX_CHAR_LENGTH)
    test_examples=process_data(testData,"test",word_counter,char_counter)
    test_tweet_idxs,test_tweet_char_idxs=build_features(test_examples,"test",word2idx_dict,char2idx_dict,MAX_SEQ_LENGTH,MAX_CHAR_LENGTH)

    # split train into train and val
    index_list=[i for i in range(len(tweet_idxs))]
    train_y=trainData["label"].values
    train_index,val_index,train_y,val_y=train_test_split(index_list,train_y,random_state=42,shuffle=True,test_size=0.2,stratify=train_y)
    tweet_idxs=np.array(tweet_idxs)
    tweet_char_idxs=np.array(tweet_char_idxs)
    train_tweet_idxs,val_tweet_idxs=tweet_idxs[train_index],tweet_idxs[val_index]
    train_tweet_char_idxs,val_tweet_char_idxs=tweet_char_idxs[train_index],tweet_char_idxs[val_index]

    #save file
    save(f"{save_dir}/word_emb.json",word_emb_mat,message="Save word embedding")
    save(f"{save_dir}/char_emb.json", char_emb_mat, message="Save char embedding")
    save(f"{save_dir}/word2idx.json", word2idx_dict, message="Save word index")
    save(f"{save_dir}/char2idx.json", char2idx_dict, message="Save char index")

    np.savez(f"{save_dir}/train_data.npz",tweet_idxs=train_tweet_idxs,tweet_char_idxs=train_tweet_char_idxs,y=train_y)
    np.savez(f"{save_dir}/val_data.npz",tweet_idxs=val_tweet_idxs,tweet_char_idxs=val_tweet_char_idxs,y=val_y)
    np.savez(f"{save_dir}/test_data.npz",tweet_idxs=np.array(test_tweet_idxs),tweet_char_idxs=np.array(test_tweet_char_idxs))









if __name__=="__main__":
    # Import spacy language model
    nlp = spacy.blank("en")
    train_file_path='../data/train_E6oV3lV.csv'
    test_file_path="../data/test_tweets_anuFYb8.csv"
    glove_file_path="../data/glove.twitter.27B/glove.twitter.27B.100d.txt"
    save_dir="../data"

    process_data(train_file_path,test_file_path,glove_file_path,save_dir)
