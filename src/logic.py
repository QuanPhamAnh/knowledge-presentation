import pandas as pd
import torch
import scipy.sparse
from transformers import RobertaTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import re
import os
import numpy as np
import torch.nn as nn
from rapidfuzz.fuzz import ratio as fuzz
from src.preprocessing import convert_accented_vietnamese_text


MODEL_PATH = "metadata_model_version_2"
device = "cpu"
def clean_metadata(sentence: str):
    if not sentence:
      return "none"
    sentence = convert_accented_vietnamese_text(sentence).lower()
    sentence = re.sub('\s+', ' ', sentence).strip()
    return sentence

def get_tokenizer():
    return RobertaTokenizerFast.from_pretrained(MODEL_PATH, max_len=512)


def get_model():
    model = torch.jit.load(
        os.path.join(MODEL_PATH, "traced_bert_embedding_sentence.pt"),
        map_location=device,
    )

    model.to(device)
    model.eval()
    return model

class MetadataEmbedding:
    def __init__(self, **kwargs) -> None:
        self.model = get_model()
        self.tokenizer = get_tokenizer()

    def get_embedding_sentences(self, sentences: List[str]) -> np.ndarray:
        sentences = list(map(clean_metadata, sentences))
        try:
            encoded_input = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
        except RuntimeError:
            self.tokenizer = get_tokenizer()
            encoded_input = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

        with torch.no_grad():
            context = self.model(**encoded_input).numpy()
        return context

class QueryResult():
  def __init__(self):
    self.meta_model = MetadataEmbedding()
    self.tracuu = pd.read_csv("data/fulldata.csv")
    self.embedding_dn = scipy.sparse.load_npz('data/embedding_dn.npz')
    self.embedding_kw = scipy.sparse.load_npz('data/embedding_kw.npz')
    all_keywords = open('data/keyphrase.txt','r')
    self.all_keywords = [i.replace("\n","") for i in all_keywords.readlines()]
    self.nd_cleantext = pd.read_csv("data/noidung_cleantext.csv")["Nội dung chi tiết"].values

  def get_sub_result(self, query: str):
      sub_index = []
      for text in re.findall('|'.join(self.all_keywords), clean_metadata(query)):
          small_df = list(self.tracuu[self.tracuu['Keyphrase'].str.contains(text)].index)
          sub_index+=small_df
      sub_index = list(set(sub_index))
      if len(sub_index)<1:
          sub_index = len(self.tracuu)
      return sub_index

  def filter_text(self, query):
    index_found = []
    clean_query = clean_metadata(query)
    for index, i in enumerate(self.nd_cleantext):
      if clean_query in i:
        index_found.append(index)
      else:
        if fuzz(i, clean_query)>95:
          index_found.append(index)
    return index_found

  def get_final_result(self, query: str):
    # filter keyword
    sub_result = self.get_sub_result(query)
    small_embedding = self.embedding_kw[sub_result[0]]
    for i in sub_result[1:]:
        small_embedding = scipy.sparse.vstack((small_embedding, self.embedding_dn[i]))

    # choose most matching defining
    embedding_query = self.meta_model.get_embedding_sentences([query])
    cos_dn = cosine_similarity(embedding_query, small_embedding)[0]
    chosen_index = np.argsort(cos_dn)[::-1][:10]

    # filter noidung
    chosen_index_noidung = self.filter_text(query)
    if len(chosen_index_noidung)>0:
      chosen_index = chosen_index_noidung

    # most matching text
    df_result = self.tracuu.iloc[[sub_result[i] for i in chosen_index]]
    small_noidung_embedding = self.embedding_dn[chosen_index[0]]
    for i in chosen_index[1:]:
        small_noidung_embedding = scipy.sparse.vstack((small_noidung_embedding, self.embedding_dn[i]))

    cos_dn = cosine_similarity(embedding_query, small_noidung_embedding)[0]
    if max(cos_dn)<0.8 and len(chosen_index_noidung)<1:
      df = df_result[['Điều – khoản tương ứng']]
    else:
      chosen_index = np.argsort(cos_dn)[::-1][0]
      df = df_result[["Điều – khoản tương ứng",	"Nội dung chi tiết"]][chosen_index:chosen_index+1]

    # final_result
    if len(df.columns)==2:
      rule, noidung = df.values[0]
      return f"Theo {rule.strip()}:\n{noidung}"
    else:
      text = "Hệ thống không chắc về kết quả bạn đã tra cứu, nhưng chúng tôi đoán rằng nó là nội dung các điều khoản:\n"
      values = [i.strip() for i in np.concatenate(df.values, axis=0)]
      values = list(set(values))
      for i in values:
        text+=f"{i}\n"
      return text