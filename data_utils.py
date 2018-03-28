import os
import sys
import urllib
import tarfile
import zipfile
from pathlib import Path
import tensorflow as tf
import shutil
import random
import re
import json

DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
file_lists = {"train":{"neg":[], "pos":[]}, "test":{"neg":[],"pos":[]}}
EMPTY_EMBEDDING = [0.0] * 50
FIXED_STRING_LENGTH = 100

def download_data():
    dest_directory = "data"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # download IMDB dataset
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'aclImdb')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    # download GLOVE embeddings
    filename = GLOVE_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(GLOVE_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_path = Path('data/glove.6B.50d.txt')
    if not extracted_path.is_file():
        print(filepath, "extracting")
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()


def get_word_embeddings(embedding_size = 50):
    global EMPTY_EMBEDDING
    EMPTY_EMBEDDING = [0.0] * embedding_size

    # first embedding is the unknown token
    vocabulary = ["--UNK--"]
    embeddings = [EMPTY_EMBEDDING]
    print("Loading embeddings to memory")
    with open("data/glove.6B.{}d.txt".format(embedding_size), "r", errors="ignore") as f:
        for line in f:
            v, e = line.split(" ", 1)
            e = list(map(float, e.split()))
            vocabulary.append(v)
            embeddings.append(e)

    print("Loaded {} embeddings".format(len(embeddings)))
    print("first: ", vocabulary[1], embeddings[1])
    return vocabulary, embeddings


def cache_everything(vocabulary, is_eval):
    is_eval_string = "test" if is_eval else "train"

    Path('data/aclImdb/{}cache/neg'.format(is_eval_string)).mkdir(parents=True, exist_ok=True)
    Path('data/aclImdb/{}cache/pos'.format(is_eval_string)).mkdir(parents=True, exist_ok=True)

    file_lists= list(Path('data/aclImdb/{}/neg'.format(is_eval_string)).glob("*")) + \
                list(Path('data/aclImdb/{}/pos'.format(is_eval_string)).glob("*"))

    for ind, filename in enumerate(file_lists):
        # check for cache
        cache_filename = Path(str(filename).replace(is_eval_string, is_eval_string + "cache"))
        if not cache_filename.exists():
            print("Caching {}/{}: {}".format(ind, len(file_lists), filename))

            with filename.open(errors="ignore") as f:
                text = f.read().split()

                word_ids, converted_text = convert_sentence(vocabulary, text)

                with cache_filename.open(mode="w", errors="ignore") as cf:
                    json.dump({"word_ids": word_ids, "converted_text": converted_text}, cf)
    print("Everything {} Cached".format(is_eval_string))


def get_sentence_batch(vocabulary, batch_size, is_eval):
    is_eval_string = "test" if is_eval else "train"
    Path('data/aclImdb/{}cache/neg'.format(is_eval_string)).mkdir(parents=True, exist_ok=True)
    Path('data/aclImdb/{}cache/pos'.format(is_eval_string)).mkdir(parents=True, exist_ok=True)

    neg_size = batch_size//2
    pos_size = batch_size - neg_size

    batch_data_filenames = []

    # collect filenames once
    if not file_lists[is_eval_string]["neg"]:
        file_lists[is_eval_string]["neg"] = list(Path('data/aclImdb/{}/neg'.format(is_eval_string)).glob("*"))
        file_lists[is_eval_string]["pos"] = list(Path('data/aclImdb/{}/pos'.format(is_eval_string)).glob("*"))

        print("found {} {}ing examples".format(
            len(file_lists[is_eval_string]["neg"])+
            len(file_lists[is_eval_string]["pos"]),
            is_eval_string))

    # randomly select negative and positive files
    batch_data_filenames.extend(random.sample(file_lists[is_eval_string]["neg"], neg_size))
    batch_data_filenames.extend(random.sample(file_lists[is_eval_string]["pos"], pos_size))

    batch_data = []
    batch_data_strings = []
    for filename in batch_data_filenames:
        # check for cache
        cache_filename = Path(str(filename).replace(is_eval_string, is_eval_string + "cache"))
        if cache_filename.exists():
            with cache_filename.open(errors="ignore") as f:
                data = json.load(f)
                batch_data.append(data["word_ids"])
                batch_data_strings.append(data["converted_text"])
        else:
            with filename.open(errors="ignore") as f:
                text = f.read()

                word_ids, converted_text = convert_sentence(vocabulary, text)

                batch_data.append(word_ids)
                batch_data_strings.append(converted_text)

                with cache_filename.open(mode="w", errors="ignore") as cf:
                    json.dump({"word_ids":word_ids, "converted_text":converted_text}, cf)

    batch_labels = [[0.0, 1.0] if x < neg_size else [1.0, 0.0]
                    for x in range(batch_size)]

    return batch_data, batch_labels, batch_data_strings


def convert_sentence(vocabulary, sentence):
    # imdb uses br for new lines
    sentence = sentence.replace("<br>", " ").replace("<br />", " ").split()

    word_ids = []
    string = []
    for ind, word in enumerate(sentence):
        word = word.lower()
        if ind >= FIXED_STRING_LENGTH:
            break

        # fast way to check if word has a letter in it
        if re.search('[a-z]', word):
            # if word has letters remove all other chars
            word = re.sub(r'[^a-z]', '', word)

        if word in vocabulary:
            word_ids.append(vocabulary.index(word))
            string.append(word)
        else:
            word_ids.append(0)
            string.append("UNK({})".format(word))

    # pad up to length with 0
    word_ids.extend([0] * (FIXED_STRING_LENGTH - len(word_ids)))
    return word_ids, " ".join(string)

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


def delete_directory(foldername):
    if os.path.exists(foldername) and os.path.isdir(foldername):
        print("Clearing folder {}".format(foldername))
        shutil.rmtree(foldername)