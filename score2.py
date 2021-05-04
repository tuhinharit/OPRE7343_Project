import datetime
import itertools
import os
import sys
from multiprocessing import Pool, freeze_support
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def line_counter(a_file):
    """Count the number of lines in a text file
    
    Arguments:
        a_file {str or Path} -- input text file
    
    Returns:
        int -- number of lines in the file
    """
    n_lines = 0
    with open(a_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def file_to_list(a_file):
    """Read a text file to a list, each line is an element
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Returns:
        [str] -- list of lines in the input file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(list, a_file, validate=True):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed
    
    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, "w", 8192000, encoding="utf-8", newline="\n") as f:
        for e in list:
            e = str(e).replace("\n", " ").replace("\r", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(list)


def read_large_file(a_file, block_size=10000):
    """A generator to read text files into blocks
    Usage: 
    for block in read_large_file(filename):
        do_something(block)
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Keyword Arguments:
        block_size {int} -- [number of lines in a block] (default: {10000})
    """
    block = []
    with open(a_file) as file_handler:
        for line in file_handler:
            block.append(line)
            if len(block) == block_size:
                yield block
                block = []
    # yield the last block
    if block:
        yield block

""""""""""""""""""""""""""""""""""""""""""""

import itertools
import math
import os
import pickle
import statistics as s
from collections import Counter, OrderedDict, defaultdict
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter

import gensim
import numpy as np
import pandas as pd
import tqdm
from sklearn import preprocessing


def expand_words_dimension_mean(
    word2vec_model,
    seed_words,
    n=50,
    restrict=None,
    min_similarity=0,
    filter_word_set=None,
):
    """For each dimensional mean vector, search for the closest n words

    
    Arguments:
        word2vec_model {gensim.models.word2vec} -- a gensim word2vec model
        seed_words {dict[str, list]} -- seed word dict of {dimension: [words]}
    
    Keyword Arguments:
        n {int} -- number of expanded words in each dimension (default: {50})
        restrict {float} -- whether to restrict the search to a fraction of most frequent words in vocab (default: {None})
        min_similarity {int} -- minimum cosine similarity to the seeds for a word to be included (default: {0})
        filter_word_set {set} -- do not include the words in this set to the expanded dictionary (default: {None})
    
    Returns:
        dict[str, set] -- expanded words, a dict of {dimension: set([words])}
    """
    vocab_number = len(word2vec_model.wv.vocab)
    expanded_words = {}
    all_seeds = set()
    for dim in seed_words.keys():
        all_seeds.update(seed_words[dim])
    if restrict != None:
        restrict = int(vocab_number * restrict)
    for dimension in seed_words:
        dimension_words = [
            word for word in seed_words[dimension] if word in word2vec_model.wv.vocab
        ]
        if len(dimension_words) > 0:
            similar_words = [
                pair[0]
                for pair in word2vec_model.wv.most_similar(
                    dimension_words, topn=n, restrict_vocab=restrict
                )
                if pair[1] >= min_similarity and pair[0] not in all_seeds
            ]
        else:
            similar_words = []
        if filter_word_set is not None:
            similar_words = [x for x in similar_words if x not in filter_word_set]
        similar_words = [
            x for x in similar_words if "[ner:" not in x
        ]  # filter out NERs
        expanded_words[dimension] = similar_words
    for dim in expanded_words.keys():
        expanded_words[dim] = expanded_words[dim] + seed_words[dim]
    for d, i in expanded_words.items():
        expanded_words[d] = set(i)
    return expanded_words


def rank_by_sim(expanded_words, seed_words, model) -> "dict[str: list]":
    """ Rank each dim in a dictionary based on similarity to the seend words mean
    Returns: expanded_words_sorted {dict[str:list]}
    """
    expanded_words_sorted = dict()
    for dimension in expanded_words.keys():
        dimension_seed_words = [
            word for word in seed_words[dimension] if word in model.wv.vocab
        ]
        similarity_dict = dict()
        for w in expanded_words[dimension]:
            if w in model.wv.vocab:
                similarity_dict[w] = model.wv.n_similarity(dimension_seed_words, [w])
            else:
                # print(w + "is not in w2v model")
                pass
        sorted_similarity_dict = sorted(
            similarity_dict.items(), key=itemgetter(1), reverse=True
        )
        sorted_similarity_list = [x[0] for x in sorted_similarity_dict]
        expanded_words_sorted[dimension] = sorted_similarity_list
    return expanded_words_sorted


def write_dict_to_csv(culture_dict, file_name):
    """write the expanded dictionary to a csv file, each dimension is a column, the header includes dimension names
    
    Arguments:
        culture_dict {dict[str, list[str]]} -- an expanded dictionary {dimension: [words]}
        file_name {str} -- where to save the csv file?
    """
    pd.DataFrame.from_dict(culture_dict, orient="index").transpose().to_csv(
        file_name, index=None
    )


def read_dict_from_csv(file_name):
    """Read culture dict from a csv file

    Arguments:
        file_name {str} -- expanded dictionary file
    
    Returns:
        culture_dict {dict{str: set(str)}} -- a culture dict, dim name as key, set of expanded words as value
        all_dict_words {set(str)} -- a set of all words in the dict
    """
    print("Importing dict: {}".format(file_name))
    culture_dict_df = pd.read_csv(file_name, index_col=None)
    culture_dict = culture_dict_df.to_dict("list")
    for k in culture_dict.keys():
        culture_dict[k] = set([x for x in culture_dict[k] if x == x])  # remove nan

    all_dict_words = set()
    for key in culture_dict:
        all_dict_words |= culture_dict[key]

    for dim in culture_dict.keys():
        print("Number of words in {} dimension: {}".format(dim, len(culture_dict[dim])))

    return culture_dict, all_dict_words


def deduplicate_keywords(word2vec_model, expanded_words, seed_words):
    """
    If a word cross-loads, choose the most similar dimension. Return a deduplicated dict. 
    """
    word_counter = Counter()

    for dimension in expanded_words:
        word_counter.update(list(expanded_words[dimension]))
    for dimension in seed_words:
        for w in seed_words[dimension]:
            if w not in word2vec_model.wv.vocab:
                seed_words[dimension].remove(w)

    word_counter = {k: v for k, v in word_counter.items() if v > 1}  # duplicated words
    dup_words = set(word_counter.keys())
    for dimension in expanded_words:
        expanded_words[dimension] = expanded_words[dimension].difference(dup_words)

    for word in list(dup_words):
        sim_w_dim = {}
        for dimension in expanded_words:
            dimension_seed_words = [
                word
                for word in seed_words[dimension]
                if word in word2vec_model.wv.vocab
            ]
            # sim_w_dim[dimension] = max([word2vec_model.wv.n_similarity([word], [x]) for x in seed_words[dimension]] )
            sim_w_dim[dimension] = word2vec_model.wv.n_similarity(
                dimension_seed_words, [word]
            )
        max_dim = max(sim_w_dim, key=sim_w_dim.get)
        expanded_words[max_dim].add(word)

    for dimension in expanded_words:
        expanded_words[dimension] = sorted(expanded_words[dimension])

    return expanded_words


def score_one_document_tf(document, expanded_words, list_of_list=False):
    """score a single document using term freq, the dimensions are sorted alphabetically
    
    Arguments:
        document {str} -- a document
        expanded_words {dict[str, set(str)]} -- an expanded dictionary
    
    Keyword Arguments:
        list_of_list {bool} -- whether the document is splitted (default: {False})
    
    Returns:
        [int] -- a list of : dim1, dim2, ... , document_length
    """
    if list_of_list is False:
        document = document.split()
    dimension_count = OrderedDict()
    for dimension in expanded_words:
        dimension_count[dimension] = 0
    c = Counter(document)
    for pair in c.items():
        for dimension, words in expanded_words.items():
            if pair[0] in words:
                dimension_count[dimension] += pair[1]
    # use ordereddict to maintain order of count for each dimension
    dimension_count = OrderedDict(sorted(dimension_count.items(), key=lambda t: t[0]))
    result = list(dimension_count.values())
    result.append(len(document))
    return result


def score_tf1(documents, document_ids, expanded_words, n_core=1):
    """score using term freq for documents, the dimensions are sorted alphabetically
    
    Arguments:
        documents {[str]} -- list of documents
        document_ids {[str]} -- list of document IDs
        expanded_words {dict[str, set(str)]} -- dictionary for scoring
    
    Keyword Arguments:
        n_core {int} -- number of CPU cores (default: {1})
    
    Returns:
        pandas.DataFrame -- a dataframe with columns: Doc_ID, dim1, dim2, ..., document_length
    """
    if n_core > 1:
        pool = Pool(n_core)  # number of processes
        count_one_document_partial = partial(
            score_one_document_tf, expanded_words=expanded_words, list_of_list=False
        )
        results = list(pool.map(count_one_document_partial, documents))
        pool.close()
    else:
        results = []
        for i, doc in enumerate(documents):
            results.append(
                score_one_document_tf(doc, expanded_words, list_of_list=False)
            )
    df = pd.DataFrame(
        results, columns=sorted(list(expanded_words.keys())) + ["document_length"]
    )
    df["Doc_ID"] = document_ids
    return df


def score_tf_idf1(
    documents,
    document_ids,
    expanded_words,
    df_dict,
    N_doc,
    method="TFIDF",
    word_weights=None,
    normalize=False,
):
    """Calculate tf-idf score for documents

    Arguments:
        documents {[str]} -- list of documents (strings)
        document_ids {[str]} -- list of document ids
        expanded_words {{dim: set(str)}}} -- dictionary
        df_dict {{str: int}} -- a dict of {word:freq} that provides document frequencey of words
        N_doc {int} -- number of documents

    Keyword Arguments:
        method {str} -- 
            TFIDF: conventional tf-idf 
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict 
            (default: {TFIDF})
        normalize {bool} -- normalized the L2 norm to one for each document (default: {False})
        word_weights {{word:weight}} -- a dictionary of word weights (e.g. similarity weights) (default: None)

    Returns:
        [df] -- a dataframe with columns: Doc_ID, dim1, dim2, ..., document_length
        [contribution] -- a dict of total contribution (sum of scores in the corpus) for each word 
    """
    print("Scoring using {}".format(method))
    contribution = defaultdict(int)
    results = []
    for i, doc in enumerate(tqdm(documents)):
        document = doc.split()
        dimension_count = OrderedDict()
        for dimension in expanded_words:
            dimension_count[dimension] = 0
        c = Counter(document)
        for pair in c.items():
            for dimension, words in expanded_words.items():
                if pair[0] in words:
                    if method == "WFIDF":
                        w_ij = (1 + math.log(pair[1])) * math.log(
                            N_doc / df_dict[pair[0]]
                        )
                    elif method == "TFIDF":
                        w_ij = pair[1] * math.log(N_doc / df_dict[pair[0]])
                    elif method == "TFIDF+SIMWEIGHT":
                        w_ij = (
                            pair[1]
                            * word_weights[pair[0]]
                            * math.log(N_doc / df_dict[pair[0]])
                        )
                    elif method == "WFIDF+SIMWEIGHT":
                        w_ij = (
                            (1 + math.log(pair[1]))
                            * word_weights[pair[0]]
                            * math.log(N_doc / df_dict[pair[0]])
                        )
                    else:
                        raise Exception(
                            "The method can only be TFIDF, WFIDF, TFIDF+SIMWEIGHT, or WFIDF+SIMWEIGHT"
                        )
                    dimension_count[dimension] += w_ij
                    contribution[pair[0]] += w_ij / len(document)
        dimension_count = OrderedDict(
            sorted(dimension_count.items(), key=lambda t: t[0])
        )
        result = list(dimension_count.values())
        result.append(len(document))
        results.append(result)
    results = np.array(results)
    # normalize the length of tf-idf vector
    if normalize:
        results[:, : len(expanded_words.keys())] = preprocessing.normalize(
            results[:, : len(expanded_words.keys())]
        )
    df = pd.DataFrame(
        results, columns=sorted(list(expanded_words.keys())) + ["document_length"]
    )
    df["Doc_ID"] = document_ids
    return df, contribution


def compute_word_sim_weights(file_name):
    """Compute word weights in each dimension.
    Default weight is 1/ln(1+rank). For example, 1st word in each dim has weight 1.44,
    10th word has weight 0.41, 100th word has weigh 0.21.
    
    Arguments:
        file_name {str} -- expanded dictionary file
    
    Returns:
        sim_weights {{word:weight}} -- a dictionary of word weights
    """
    culture_dict_df = pd.read_csv(file_name, index_col=None)
    culture_dict = culture_dict_df.to_dict("list")
    sim_weights = {}
    for k in culture_dict.keys():
        culture_dict[k] = [x for x in culture_dict[k] if x == x]  # remove nan
    for key in culture_dict:
        for i, w in enumerate(culture_dict[key]):
            sim_weights[w] = 1 / math.log(1 + 1 + i)
    return sim_weights






"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Hardware options
N_CORES: int = 6  # max number of CPU cores to use
RAM_CORENLP: str = "8G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 100 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

# Directory locations
os.environ[
    "CORENLP_HOME"
] = "C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Semesters/Sem 4 - Spring 2021/Modern Mc Learning - OPRE 7343/Assgn/OPRE7343_Project/stanford-corenlp-4.2.0/"  # location of the CoreNLP models; use / to seperate folders
DATA_FOLDER: str = "C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Semesters/Sem 4 - Spring 2021/Modern Mc Learning - OPRE 7343/Assgn/OPRE7343_Project/data/"
MODEL_FOLDER: str = "C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Semesters/Sem 4 - Spring 2021/Modern Mc Learning - OPRE 7343/Assgn/OPRE7343_Project/models/" # will be created if does not exist
OUTPUT_FOLDER: str = "C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Semesters/Sem 4 - Spring 2021/Modern Mc Learning - OPRE 7343/Assgn/OPRE7343_Project/outputs/" # will be created if does not exist; !!! WARNING: existing files will be removed !!!

# Parsing and analysis options
STOPWORDS: Set[str] = set(
    Path("C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Semesters/Sem 4 - Spring 2021/Modern Mc Learning - OPRE 7343/Assgn/OPRE7343_Project/resources", "StopWords_Generic.txt").read_text().lower().split()
)  # Set of stopwords from https://sraf.nd.edu/textual-analysis/resources/#StopWords
PHRASE_THRESHOLD: int = 10  # threshold of the phraser module (smaller -> more phrases)
PHRASE_MIN_COUNT: int = 10  # min number of times a bigram needs to appear in the corpus to be considered as a phrase
W2V_DIM: int = 300  # dimension of word2vec vectors
W2V_WINDOW: int = 5  # window size in word2vec
W2V_ITER: int = 20  # number of iterations in word2vec
N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
DICT_RESTRICT_VOCAB = None # change to a fraction number (e.g. 0.2) to restrict the dictionary vocab in the top 20% of most frequent vocab

# Inputs for constructing the expanded dictionary
DIMS: List[str] = ["integrity", "teamwork", "greeninnovation", "respect", "quality"]
SEED_WORDS: Dict[str, List[str]] = {
    "integrity": [
        "integrity",
        "ethic",
        "ethical",
        "accountable",
        "accountability",
        "trust",
        "honesty",
        "honest",
        "honestly",
        "fairness",
        "responsibility",
        "responsible",
        "transparency",
        "transparent",
    ],
    "teamwork": [
        "teamwork",
        "collaboration",
        "collaborate",
        "collaborative",
        "cooperation",
        "cooperate",
        "cooperative",
    ],
    "greeninnovation": [
        "innovation",
        "sustainable",
        "research",
        "energy",
        "environmental",
        "social",
        "development",
        "technology",
        "sustainability",
        "economy",
        "efficiency",
        "technologies",
        "green",
        "impact",
        "innovations",
        "technological",
        "transitions",
    ],
    "respect": [
        "respectful",
        "talent",
        "talented",
        "employee",
        "dignity",
        "empowerment",
        "empower",
    ],
    "quality": [
        "quality",
        "customer",
        "customer_commitment",
        "dedication",
        "dedicated",
        "dedicate",
        "customer_expectation",
    ],
}


# Create directories if not exist
Path(DATA_FOLDER, "processed", "parsed").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "unigram").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "bigram").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "trigram").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "w2v").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "dict").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores", "temp").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=True, exist_ok=True)

import itertools
import os
import pickle
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import pandas as pd
from tqdm import tqdm as tqdm

#import culture.culture_dictionary
#import culture.file_util

#os.chdir(r"C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Semesters/Sem 4 - Spring 2021/Modern Mc Learning - OPRE 7343/Assgn/OPRE7343_Project")
#os.getcwd()
# @TODO: The scoring functions are not memory friendly. The entire pocessed corpus needs to fit in the RAM. Rewrite a memory friendly version.


def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """Construct document level corpus from sentence level corpus and write to disk.
    Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(global_options.OUTPUT_FOLDER, "scores", "temp"). 
    
    Arguments:
        sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file {str or Path} -- The sentence ID file, each line correspond to a line in the sent_co(docID_sentenceID)
    
    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Constructing doc level corpus")
    # sentence level corpus
    sent_corpus = file_to_list(sent_corpus_file)
    sent_IDs = file_to_list(sent_id_file)
    assert len(sent_IDs) == len(sent_corpus)
    # doc id for each sentence
    doc_ids = [x.split("_")[0] for x in sent_IDs]
    # concat all text from the same doc
    id_doc_dict = defaultdict(lambda: "")
    for i, id in enumerate(doc_ids):
        id_doc_dict[id] += " " + sent_corpus[i]
    # create doc level corpus
    corpus = list(id_doc_dict.values())
    doc_ids = list(id_doc_dict.keys())
    assert len(corpus) == len(doc_ids)
    with open(
        Path(OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
        "wb",
    ) as out_f:
        pickle.dump(corpus, out_f)
    with open(
        Path(OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "wb"
    ) as out_f:
        pickle.dump(doc_ids, out_f)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def calculate_df(corpus):
    """Calcualte and dump a document-freq dict for all the words.
    
    Arguments:
        corpus {[str]} -- a list of documents
    
    Returns:
        {dict[str: int]} -- document freq for each word
    """
    print("Calculating document frequencies.")
    # document frequency
    df_dict = defaultdict(int)
    for doc in tqdm(corpus):
        doc_splited = doc.split()
        words_in_doc = set(doc_splited)
        for word in words_in_doc:
            df_dict[word] += 1
    # save df dict
    with open(
        Path(OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle"), "wb"
    ) as f:
        pickle.dump(df_dict, f)
    return df_dict


def load_doc_level_corpus():
    """load the corpus constructed by construct_doc_level_corpus()
    
    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Loading document level corpus.")
    with open(
        Path(OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
        "rb",
    ) as in_f:
        corpus = pickle.load(in_f)
    with open(
        Path(OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "rb"
    ) as in_f:
        doc_ids = pickle.load(in_f)
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def score_tf(documents, document_ids, expanded_words):
    """
    Score documents using term freq. 
    """
    print("Scoring using Term-freq (tf).")
    score = score_tf1(
        documents=documents,
        document_ids=document_ids,
        expanded_words=expanded_words,
        n_core=N_CORES,
    )
    score.to_csv(
        Path(OUTPUT_FOLDER, "scores", "scores_TF.csv"), index=False
    )


def score_tf_idf(documents, doc_ids, N_doc, method, expanded_dict, **kwargs):
    """Score documents using tf-idf and its variations
    
    Arguments:
        documents {[str]} -- list of documents
        doc_ids {[str]} -- list of document IDs
        N_doc {int} -- number of documents
        method {str} -- 
            TFIDF: conventional tf-idf 
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict
        expanded_dict {dict[str, set(str)]} -- expanded dictionary
    """
    if method == "TF":
        print("Scoring TF.")
        score_tf(documents, doc_ids, expanded_dict)
    else:
        print("Scoring TF-IDF.")
        # load document freq
        df_dict = pd.read_pickle(
            Path(OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
        )
        # score tf-idf
        score, contribution = score_tf_idf1(
            documents=documents,
            document_ids=doc_ids,
            expanded_words=expanded_dict,
            df_dict=df_dict,
            N_doc=N_doc,
            method=method,
            **kwargs
        )
        # save the document level scores (without dividing by doc length)
        score.to_csv(
            str(
                Path(
                    OUTPUT_FOLDER,
                    "scores",
                    "scores_{}.csv".format(method),
                )
            ),
            index=False,
        )
        # save word contributions
        pd.DataFrame.from_dict(contribution, orient="index").to_csv(
            Path(
                OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                "word_contribution_{}.csv".format(method),
            )
        )


if __name__ == "__main__":
    current_dict_path = str(
        str(Path(OUTPUT_FOLDER, "dict", "expanded_dict.csv"))
    )
    culture_dict, all_dict_words = read_dict_from_csv(
        current_dict_path
    )
    # words weighted by similarity rank (optional)
    word_sim_weights = compute_word_sim_weights(current_dict_path)

    ## Pre-score ===========================
    # aggregate processed sentences to documents
    corpus, doc_ids, N_doc = construct_doc_level_corpus(
        sent_corpus_file=Path(
            DATA_FOLDER, "processed", "trigram", "documents.txt"
        ),
        sent_id_file=Path(
            DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
        ),
    )
    word_doc_freq = calculate_df(corpus)

    ## Score ========================
    # create document scores
    methods = ["TF", "TFIDF", "WFIDF"]
    for method in methods:
        score_tf_idf(
            corpus,
            doc_ids,
            N_doc,
            method=method,
            expanded_dict=culture_dict,
            normalize=False,
            word_weights=word_sim_weights,
        )

