"download transcripts"

import sys
sys.modules[__name__].__dict__.clear()

import os
os.chdir(r'C:\Users\txh180026\OneDrive - The University of Texas at Dallas\work\Research\Climate change\Proxy Voting\Code\Step2_Innovation Textual Analysis')
os.getcwd()
pip install wrds
import wrds
import pandas as pd
db = wrds.Connection(wrds_username='tharit')
db.create_pgpass_file()
db.close()
db = wrds.Connection(wrds_username='tharit')

db.list_libraries()
db.list_tables(library="ciq_transcripts")
des = db.describe_table(library="ciq_transcripts", table="ciqtranscript")
date = db.get_table(library='ciq_transcripts', table='ciqtranscript')
date.to_stata("transcriptid_date.dta", version=117)


data1 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', obs = 10000000)
data1.to_stata("data1.dta", version=117)
del data1

data2 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', obs = 10000000, offset= 9999999)
data2.to_stata("data2.dta", version=117)
del data2

data3 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', obs = 10000000, offset= 19999999)
data3.to_stata("data3.dta", version=117)
del data3

data4 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', obs = 10000000, offset= 29999999)
data4.to_stata("data4.dta", version=117)
del data4

data5 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', obs = 10000000, offset= 39999999)
data5.to_stata("data5.dta", version=117)
del data5

data6 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', obs = 10000000, offset= 49999999)
data6.to_stata("data6.dta", version=117)
del data6

data7 = db.get_table(library='ciq_transcripts', table='ciqtranscriptcomponent', offset= 59999999)
data7.to_stata("data7.dta", version=117)
del data7





"Textual analysis"
file = pd.read_excel("transcripts_final.xlsx")


import csv
import glob
import re
import string
import sys
import time
import Load_MasterDictionary as LM
import numpy as np

# User defined file pointer to LM dictionary
MASTER_DICTIONARY_FILE = r'C:/Users/txh180026/OneDrive - The University of Texas at Dallas/work/Research/Climate change/Proxy Voting/Code/Step1/LM_Mod_Dict_Innovation.csv'

lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, True)


for index in file.index:
    print(index)
    _odata = [0] * 6
    text = file.loc[index,"componenttext"]
    text = re.sub('(May|MAY)', ' ', text)  # drop all May month references
    text = text.upper()  # for this parse caps aren't informative so shift
    vdictionary = {}
    total_syllables = 0
    word_length = 0
    tokens = re.findall('\w+', text)  # Note that \w+ splits hyphenated words

    for token in tokens:
        if not token.isdigit() and len(token) > 1 and token in lm_dictionary:
            _odata[1] += 1  # word count
            word_length += len(token)
            if token not in vdictionary:
                vdictionary[token] = 1
            if lm_dictionary[token].green_innovation: _odata[2] += 1
            if lm_dictionary[token].innovation: _odata[3] += 1
            if lm_dictionary[token].uncertainty: _odata[4] += 1
            if lm_dictionary[token].litigious: _odata[5] += 1
            total_syllables += lm_dictionary[token].syllables
    file.loc[index,"word count"] = _odata[1]
    file.loc[index,"numeric count"] = len(re.findall('[0-9]', text))
    file.loc[index,"syllable"] = total_syllables
    file.loc[index,"word length"] = word_length
    file.loc[index,"green innovation"] = _odata[2] / file.loc[index,"word count"]
    file.loc[index,"innovation"] = _odata[3] / file.loc[index,"word count"]
    file.loc[index,"uncertainity"] = _odata[4] / file.loc[index,"word count"]
    file.loc[index,"litigious"] = _odata[5] / file.loc[index,"word count"]


file.to_stata("innovate_score.dta", version=117)



