import json
import re
import string
import matplotlib.pyplot as plt
import nltk
from IPython.display import display, Markdown


def read_json(input_file):
    """Read a .json file and return a list of entries"""
    with open(input_file, "r", encoding='utf-8') as f:
        f_lines = f.readlines()
        lines = []
        for line in f_lines:
            lines.append(json.loads(line.strip()))
        return lines


def read_json_v2(input_file):
    """Read a json file of mongodb format"""
    with open(input_file, "r", encoding='utf-8') as f:
        return json.load(f)


def write_json(output_file, data):
    """Write data with appropriate format to a .json file"""
    with open(output_file, "w", encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def cache(pos=()):
    """Caching decorator to store previously returned results"""
    def decorator_wrapper(func):
        cache_dict = {}

        def wrapper(*args, **kwargs):
            end_key = tuple(args[k] for k in pos)
            if end_key not in cache_dict:
                cache_dict[end_key] = func(*args, **kwargs)
            return cache_dict[end_key]

        return wrapper
    return decorator_wrapper


def remove_numbers(text):
    return re.sub(r'[0-9]+', '', text)


def remove_accents(text):
    text = re.sub('ά', 'α', text)
    text = re.sub('έ', 'ε', text)
    text = re.sub('ή', 'η', text)
    text = re.sub('ό', 'ο', text)
    text = re.sub('ύ', 'υ', text)
    text = re.sub('ϋ', 'υ', text)
    text = re.sub('ΰ', 'υ', text)
    text = re.sub('ώ', 'ω', text)
    text = re.sub('ί', 'ι', text)
    text = re.sub('ϊ', 'ι', text)
    text = re.sub('ΐ', 'ι', text)
    return text


def pad_punctuation(text):
    """Pad punctuation with spaces and remove multiple spaces"""
    punctuation = '([' + string.punctuation + '«»“”‘’–─‹″€΄•…¨' + '])'
    text = re.sub(punctuation, r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def remove_punctuation(text):
    """Remove punctuation"""
    punctuation = '([' + string.punctuation + '«»“”‘’–─‹″€΄•…¨' + '])'
    text = re.sub(punctuation, r'', text)
    return text


def plot_graphs(history, metric, plot_val=False):
    """Plot history of keras model training for train and validation sets"""
    plt.plot(history.history[metric])
    if plot_val:
        plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    if plot_val:
        plt.legend([metric, 'val_'+metric])
    else:
        plt.legend([metric])


def print_scores(glove_model, text, scores):
    text_tokenized = nltk.word_tokenize(pad_punctuation(remove_numbers(text)))

    output = ''
    ids = glove_model.string_to_ids(text, return_none=True)
    j = 0
    for i, t in zip(ids, text_tokenized):
        if i is not None:
            output += f'<span style="background-color: rgba(0,180,0,{scores[j]})">{t}</span> '
            j += 1

    display(Markdown(output))
