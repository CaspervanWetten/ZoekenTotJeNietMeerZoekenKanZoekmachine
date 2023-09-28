import math
import pandas as pd
import numpy as np
import os
import glob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
import sys
import requests
import csv

#Declaring all needed variables
directory = r'.txt_files'
all_lists = []
tf_dictionary_list = []
uniq_dict = {}
idf_dict = {}
term_weight_list = []
tf_query_dictionary_list = []
vector_len_list = []
tot_words_in_query = 0.0
q_value = 0
compared_length_list = []
cosine_sim_dict = {}
sorted_sim_dict = {}
sorted_sim_text_list = []


#USED FUNCTIONS FROM LIBRARIES
wordnet_lemmatizer = WordNetLemmatizer()

#IMPORT INITIALIZATIONS, DO NOT TOUCH
#I know that we need this for finding the context and meaning of words for the lemmatization process, but I don't know how it works, but it does work. So pls no touchie
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB #I think it is needed to declare the word types (noun, adjective, verb, adverb) that it'll determine later?
tag_map['R'] = wn.ADV


#ALL THE TEXT OPERATIONS, FROM STOPWORD REMOVAL TO LEMMATIZATION
def RemoveSymbols(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[]^_`{|}\\~\n',"
    for i in symbols: #For every symbol that is in the string symbols, if that symbol is in the string data, replace it with nothing (thus removing it)
        data = data.replace(i, "")
    return(data)
def RemoveStopWords(text):
    tokenized_text = word_tokenize(text) #Tokenizing the text turns it from a string into a list,
    remove_sw_clean_code = [] #This name, remove_sw_clean_code was chosen because we had a different RemoveStopWords function first, but it was spaghetti so we redid it
    for word in tokenized_text: 
        if word not in stopwords.words():
            remove_sw_clean_code.append(word) #For every word in the new tokenized text, check if it's in the stopword list, if not, then add it to the new string
    return(remove_sw_clean_code)
def Lemmatization(word): #The concept of Lemmatization is to change words into their base form, e.g. "we are" -> "we be", however some words have different meanings depending on their context
    final_list = []
    for i in word:
        tokens = word_tokenize(i)
        for token, tag in pos_tag(tokens):
            final_list.append(wordnet_lemmatizer.lemmatize(token,tag_map[tag[0]])) #This direwire of a function takes care of the context bit. the "token, tag_map[tag[0]]" determines what kind of word it is in that context, and then bring them back to their base form
            #How, you ask? honestly no clue, nltk does it for us, my guess is on dark magic
    return(final_list)

#CALLS ALL THE DIFFERENT TEXT OPERATIONS IN A SWEET NICE SINGLE PACKAGE
def TextOperations(text):
    #We also had a RemoveSingleLetters function, however determined that it was unneeded, considering the fact that we shouldn't run into a situation where there are just single letters randomly flaffing about, and if it happens, something else must've gone horrible wrong
    text = text.lower()
    data =  RemoveSymbols(text)
    data =  RemoveStopWords(data)
    data =  Lemmatization(data)
    return(data)

def CalcTF(all_lists):
    for item in all_lists: #We now have the list "all_lists", which has a nested list for every document, e.g. all_lists =[ ['a','b'], ['a','c'] ], for every list, it makes a temporary dictionary, which will be added to a list as its own list item, and then emptied
        temp_list_dict = {}
        for i in item: #So for every word in the nested list
            if i in temp_list_dict: #If I, where I is the word in the nested list, is already in the temporary dict, add +1, else, add it and give it a value of 1
                temp_list_dict[i] += 1
            else:
                temp_list_dict[i] = 1
        tf_dictionary_list.append(temp_list_dict) #To reitarate, temp_list_dict is a dictionary that holds the data we need it to, gets added to a list and then is emptied again to do it for the next nested list. This approach allows us to have X amount of documents (where X approaches infinite) in the final list. This is much more dynamic and robust then, e.g. adding "word_dict_a"
    return(tf_dictionary_list)
def CalcIDF(word_list):
    for list in word_list: #The inverse document frequency counts in how many documents a term appears
        uniq_list = [] #We make this temporary list, we use this lsit to check if a term has already appeared earlier in the text, if it has, do not add it to the dictionary again, this way we get only all the unique words
        for item in list:
            if item not in uniq_list: #If the item is in the list, do not add it to the dict again
                if item not in uniq_dict:
                    uniq_dict[item] = 1
                else:
                    uniq_dict[item] += 1
                uniq_list.append(item)
        #We now have a dictionary with every unique item in the Nth list of word_list
        document_amount = len(word_list)
        for word, val in uniq_dict.items():
            idf_dict[word] = math.log2(document_amount/(float(val))) #do the calculations
    return(idf_dict)
def CalcTermWeight(tf_dictionary_list, idf_dict): 
    for dict in tf_dictionary_list:
        temp_weight_dict = {}
        for word, val in dict.items():
            temp_weight_dict[word] = val*idf_dict[word] #The term weight matrix is simply the frequency of a term in a document, * their respective idf, so thats what we do here
        term_weight_list.append(temp_weight_dict)
    return(term_weight_list)

def FullTermWeight(data):
    #This calls all the the other needed TF-IDF functions in one neat little package
    tf_dict_list = CalcTF(data)
    idf_list = CalcIDF(tf_dict_list)
    data = CalcTermWeight(tf_dict_list, idf_list)
    return(data)

def GetTextDocus(dir):
    #This gets us all the documents and their inner texts
    for file in os.listdir(dir): #For every file, do the following
        f = open(os.path.join(dir, file), 'r+', encoding='utf8')
        text = f.read().replace('\n', ' ') #replace every \n (thus every linebreak) with a space character
        text = text.lower() #We lowercase everything, because Pear is a different word than pear according to PC's, but to us they're the same
        all_lists.append(TextOperations(text))  #and add the text to the list, now do it all again : D
    return(all_lists)

def ReCalcCSV(): #We use this function to simply recalculate the entire CSV file, that way we can 
    #Simply call this
    all_lists = GetTextDocus(directory)
    term_weight_matrix = FullTermWeight(all_lists)
    print(term_weight_matrix)
    csv_dataframe = pd.DataFrame(term_weight_matrix) #We make the matrix into a dataframe to make it easier to read for us as human beings
    print(csv_dataframe)
    csv_dataframe.to_csv('term_weight_matrix.csv', encoding='utf-8', index=False) #After that, the dataframe is written to a .csv file for easy reading
    return()


def GetCosineSimilarity(query): #This is the "query" function, it gets the query, does all the math and than compares the vector length
    sorted_sim_text_list = [] #We start with declaring this variable, we declare these here instead of up top, because we need to e.g. re-empty this list everytime this function is called, else, if you search multiple queries in the same session, the new results will be added BEHIND the old results thus causing issues
    tot_words_in_query = 0 
    term_weight_list = []
    query = TextOperations(query) #We also do all our TextOperations function on the query
    with open('term_weight_matrix.csv', 'r', encoding='utf8') as data: #Here, we read the term_weight matrix into a list
        for line in csv.DictReader(data):
            term_weight_list.append(line)
    
    for document in term_weight_list:
        temp_vector_len = 0
        for word, value in document.items(): #The list term_weight_list contains a bunch of dictionaries, for each of these dictionaries, grab the word (the word) and the value (so the term_weight value) 
            try: 
                value = float(value) #Because the term_weight is saved as a string, and you can't do math on a string, we turn the values into Floats
            except: 
                value = 0.0 #There are a lot of words that exist only in 1 document, or atleast aren't in _all_ documents, because the weight of these words for the documents that don't have those words are saved as NaN, undefined,
                            #And you can't turn undefined into a float, the function float(value) occasionally throws an error, however because all of these values are in practically the same as 0, we do this try: except: block
                            #Try to float(value), but if it doesn't work (so if the value is NaN), make the value a 0! (also 0 would be an int, hence the 0.0)
            temp_vector_len += np.square(value)  #We square the value, then add it to a temporary list, this list will contain all squared version of the terms.
        temp_vector_len = np.sqrt(temp_vector_len) #After we squared the values, we square root the _entire_ list
        vector_len_list.append(temp_vector_len) #Now, add the square-root version of the temp_list to the actual list 

    for i in query: #for I in query (which is now a list of words, so every I is a single word), add 1
        tot_words_in_query +=1
    query_len = np.sqrt(tot_words_in_query) #To calculate the vector length, you take the square root of the total amount of words in the query
    temp_vector_len_list = [] #We make another temporary list
    for docu_vector in vector_len_list: #for every documents vector length in the list vector_len_list (that we created above) take that document vector, and do some math on it
        temp_calc = 0.0
        for document in term_weight_list:  #We go back to the document, this document is a dictionary, with every word in that document+its term weight matrix
            q_value = 0
            for word, value in document.items(): #For every word in the document, check if it is IN query, so if the people searched for it, if its true add the value of that word to the temporary q_value variable
                for q_word in query:
                    try: 
                        value = float(value)
                    except: 
                        value  = 0.0
                    if word == q_word:
                        q_value += value
                    temp_calc = q_value/(query_len*docu_vector) #We make a temporary variable, which now has the value of the compared vector lengths of the document compared to the queries
            temp_vector_len_list.append(temp_calc) #Add that variable to this list, this list now has X amount of values (where X is the amount of documents you have) in the order that the documents are arranged in the file structure (so for us, that'd be in alphabetical order)

    temp_docu_name_list = []
    for file in os.listdir(directory): 
        f = open(os.path.join(directory, file), 'r+', encoding='utf8')
        temp_docu_name_list.append(file) #For every file in the directory, add that file to this list (so this is a list with file names)

    for i in range(len(temp_docu_name_list)): #For i in the range of length of the vector_len_list, so do this for every document you have, no more, no less
        cosine_sim_dict[temp_docu_name_list[i]] = temp_vector_len_list[i] #Make a dictionary, with the name of the file as a key, and the value of the cosine similarity as the value
    sorted_sim_dict = dict(sorted(cosine_sim_dict.items(), key=lambda item: item[1], reverse=True)) #Now sort that dictionary in descending order (so 2->1->0)
    
    for key, value in sorted_sim_dict.items(): #For Key (so the name of the text) and value (so the cosine similarity) do this
        temp_list = [] #We create a temporary list, to this list we will add three things, the name of the file (as item 1), the cosine similiarity (as item 2) and then all the words in that document (as item 3)
        temp_list.append(key) #Add the name of the text
        temp_list.append(value) #Add the cosine similarity 
        for file in os.listdir(directory): #For every file in the directory
            f = open(os.path.join(directory, file), 'r+', encoding='utf8') 
            text = f.read() #Open that file, and read that file to the variable text
            if key == file: #If the name of that key (so the name of the file) == the name of the file that they are currently at 
                temp_list.append(text) #add the text to the list
        sorted_sim_text_list.append(temp_list) #Now, append the final list with the list that you just made, so the final list, at the end, returns a list, with many nested lists, all the nested lists have the Article name, the cosine similarity to the query and lastly all the words
    return(sorted_sim_text_list)

def WriteToTxt(title, text):
    try: #Sometimes, depending on what characters one copy-pastes into the "add your own article" section, this simply fails, probably due to encoding errors, hence the try: except: block
        path_name = os.path.join(directory, title+".txt") #The name of the path, thus the thing will be written to is a combination of "directory" (thus /.txt_files) and the title (as given by the user) plus ".txt" to signify it being a .txt file
        file = open(path_name, 'w', encoding='utf8') #Try to to open this file as W (so make this file and write to it)
        file.write(text) #write the text as given by the user
        ReCalcCSV() #Recalculate the term weight matrix, so the file can actually be searched for
        return("We hebben sucessvol uw artikel toegevoegd :D")
    except:
        return("Helaas, er is iets fout gegaan, probeer het nog een keer. Als u deze error blijft krijgen, neem dan contact op met ZTJNMZK@info.nl")