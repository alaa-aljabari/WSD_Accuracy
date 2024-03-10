import re
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import pandas as pd
from ArabGlossBERT.arabert.preprocess import ArabertPreprocessor
from transformers import BertTokenizer,BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json 
from tokenizers_words import simple_word_tokenize

#print("Start")
dftrue = pd.DataFrame()
model = BertForSequenceClassification.from_pretrained('{}'.format('./ArabGlossBERT/bert-base-arabertv02_22_May_2021_00h_allglosses_unused01'),
                                                      output_hidden_states = True,
                                                      num_labels=2
                                                      )
tokenizer = BertTokenizer.from_pretrained('{}'.format('./ArabGlossBERT/bert-base-arabertv02'))


#print("Model loaded...")


def normalizearabert(s):
  model_name = 'aubmindlab/bert-base-arabertv02'
  arabert_prep = ArabertPreprocessor(model_name.split("/")[-1])
  return arabert_prep.preprocess(str(s))



def glosses1(dfcand,target):
# """
# takes a dataframe 
# return 
	# 'none' if the maximum logistic regression score for TRUE class is less than -2 OR
	# the predicted gloss having the maximum logistic regression score
# """

  wic_c = []
  wic_c, _ = read_data(dfcand,normalizearabert,target)
  tokenizedwic_c = np.array([tokenizer.encode(x, max_length=512,padding='max_length',truncation='longest_first',add_special_tokens=True) for x in wic_c])
  max_len = 512
  segmentswic = torch.tensor([get_segments(tokenizer.convert_ids_to_tokens(i),max_len) for i in tokenizedwic_c])
  paddedwic = tokenizedwic_c
  attention_maskwic = np.where(paddedwic != 0, 1, 0)
  input_idswic = torch.tensor(paddedwic)  
  attention_maskwic = torch.tensor(attention_maskwic)
  model2 = model.eval()
  wicpredictions , wictrue_labels = [], []
  b_input_ids = input_idswic
  b_input_mask =  attention_maskwic
  b_input_seg = segmentswic

  with torch.no_grad():
    outputs = model2(b_input_ids,token_type_ids=b_input_seg,attention_mask=b_input_mask)

  logits = outputs[0]
  wicpredictions.append(logits)
  wicflat_predictions = np.concatenate(wicpredictions, axis=0)

  return dfcand['Concept_id'].to_list()[np.argmax(wicflat_predictions, axis=0).flatten()[1]],dfcand['Gloss'].to_list()[np.argmax(wicflat_predictions, axis=0).flatten()[1]]


def read_data(data,normalize,target):
  c = []
  labels = []
  for i,row in data.iterrows():
      
      example = normalize(row['Example'])
      gloss = normalize(row['Gloss'])
      label = row['Label']

      
      c.append('{} [SEP] {}: {}'.format(example,target,gloss))
      if label == 1.0:
          labels.append(1)
      else:
          labels.append(0)
  return c,labels


def inserttag1(sentence,tag,start,end):
    before = sentence[:start]
    after = sentence[end:]
    target = sentence[start:end]
    return before+tag+sentence[start:end]+tag+after


def get_segments(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def senttarget(target,example):
  start = -1
  try:
    start = example.index(target)
  except ValueError:
    return -1
  end = example.index(target)+len(target)
  return inserttag1(example,"[UNUSED0]",start,end)

def GlossPredictor(target, example, glosses):
  example = senttarget(target,example)
  if example == -1:
     return -1,-1
  
  data = []
  for conceptId in glosses.keys():
      data.append([conceptId, "", "", glosses[conceptId], target, example, 0,1,'','',''])
  dfcolumns = ['Concept_id', 'Diac_lemma', 'Undiac_lemma', 'Gloss', 'Target', 'Example', 'Is_training', 'Label', 'concept_id', 'lemma_id', 'POS']
  dfcand = pd.DataFrame(data,columns=dfcolumns)
  
  
  if len(dfcand) > 0:
    dfcand['Example'] = dfcand['Example'].apply(lambda x: example)
    dfcand['Target'] = dfcand['Target'].apply(lambda x: target)
    dfcand = dfcand.drop_duplicates()
  
    dfcand['Example'] = dfcand['Example'].apply(lambda x: x.upper())
    dfcand['Example'] = dfcand['Example'].apply(lambda x: re.sub(r'^((.?\[UNUSED0\].?){1})\[UNUSED0\]', r'\1[UNUSED1]', x))
    return glosses1(dfcand,target)
  else:
    return 'none','none'


filePath = 'WSD_Accuracy/Data/Test.json' #'./ArabGlossBERT/examples.json'
glossesDict= "WSD_Accuracy/Data/dictionary.json" #"./ArabGlossBERT/dictionary.json"
glossesDictContent = {}
with open(filePath, 'r', encoding='utf-8') as file:
   fileContent = file.read()

with open(glossesDict, 'r', encoding='utf-8') as file2:
   glossesDictValues = json.loads(file2.read())

for glossesDictValue in glossesDictValues:
   glossesDictContent[glossesDictValue["sense_id"]] = glossesDictValue["definition"]

# Load the JSON data
sentencesInfo = json.loads(fileContent)
print(sentencesInfo)
    
targetWord = ""
glossesIds = [] 

def WSDdisambiguation(inputSentence, inputWord):
   for sentenceInfo in sentencesInfo:
     glossesDictionary = {}
     sentence = sentenceInfo["sentence"]
     if sentence == inputSentence: 
        for w in sentenceInfo["words"]:
            if inputWord == w["word"]:
               targetWord = w["word"]
               glossesIds = w["senses"]
        for glossId in glossesIds: 
            if glossId in glossesDictContent.keys():
               glossesDictionary[glossId] = glossesDictContent[glossId]
        conceptId, gloss = GlossPredictor(targetWord, sentence, glossesDictionary)
        return conceptId, gloss
   return 0, "Enter the valid sentence"



def WSD(sentence, targetWord): 
        i = 1 
        outputList = [] 
        j = 1
        found = False
#        print("Sentnece : ", i , " word id : ", j)
        wordsJson = []
        words = simple_word_tokenize(sentence) 
        for word in words: 
          if word == targetWord:
            found = True
            conceptId, gloss = WSDdisambiguation(sentence, word)
            wordsJson.append({
                "word_id": j,
                "word": word,
                "target_gloss": conceptId
            })
            j = j + 1
        if found == False:
           wordsJson = ["The target word wasn't found in the entered sentence."]  
        sentenceJson = {"sentence_id": i , "sentence": sentence, "words": wordsJson}
        i = i + 1
        outputList.append(sentenceJson)
        return outputList


def WSD_Evaluation(sentences_json_path, output_json_path):
    outputList = []  
    with open(sentences_json_path, 'r', encoding='utf-8') as file:
        sentences = json.load(file)
        for sentence in sentences:
            wordsJson = []
            for word in sentence['words']:
                conceptId, gloss = WSDdisambiguation(sentence['sentence'], word['word'])
                wordsJson.append({
                    "word_id": word['word_id'],
                    "word": word['word'],
                    "target_sense": conceptId  # Assuming only one sense for simplicity
                })
            
            sentenceJson = {
                "sentence_id": sentence['sentence_id'],
                "sentence": sentence['sentence'],
                "words": wordsJson
            }
            outputList.append(sentenceJson)
            
    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(outputList, outfile,ensure_ascii=False, indent=4)

    print("Output JSON file saved at:", output_json_path)
    return outputList


#print(WSD(["كيف ساهمت السياسة", "الأميركية المستندة إلى"]))
