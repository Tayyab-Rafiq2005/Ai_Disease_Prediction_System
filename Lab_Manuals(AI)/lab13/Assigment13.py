import numpy as np 
import pandas as pd  
!pip install gensim 
import gensim  
import os 
!pip install nltk  
data = "C:/Users/Syed Hamedoon/VScode Examples/data"  
import nltk  
nltk.download('punkt') 
nltk.download('punkt_tab') 
import os 
from nltk import sent_tokenize 
from gensim.utils import simple_preprocess  
DATA_PATH = r"C:\Users\Syed Hamedoon\VScode Examples\data"  
story = []  
for filename in os.listdir(DATA_PATH): 
    if filename.endswith(".txt"): 
        file_path = os.path.join(DATA_PATH, filename) 
        try: 
            with open(file_path, "r", encoding="utf-8") as f: 
                corpus = f.read() 
        except UnicodeDecodeError: 
            with open(file_path, "r", encoding="cp1252") as f: 
                corpus = f.read() 
        for sent in sent_tokenize(corpus): 
            story.append(simple_preprocess(sent)) 
print(len(story)) 
print(story[:2]) 
model = gensim.models.Word2Vec( 
    window=10, 
    min_count=2 ) 
model.build_vocab(story) 
model.train(story, total_examples=model.corpus_count, epochs=model.epochs )
model.wv.most_similar('daenerys') 
model.wv.doesnt_match(['jon','rikon','robb','arya','sansa','bran']) 
model.wv.doesnt_match(['cersei', 'jaime', 'bronn', 'tyrion']) 
model.wv['king'] 
model.wv.similarity('arya','sansa') 
model.wv.similarity('tywin','sansa') 
model.wv.get_normed_vectors() 
y = model.wv.index_to_key 
len(y)
from sklearn.decomposition import PCA 
pca = PCA(n_components=3)
X = pca.fit_transform(model.wv.get_normed_vectors()) 
!pip install --upgrade nbformat 
import pandas as pd 
import plotly.express as px 
import plotly.io as pio 
pio.renderers.default = "browser"  # ← IMPORTANT 
df = pd.DataFrame(X[200:300], columns=["x", "y", "z"]) 
df["label"] = y[200:300] 
fig = px.scatter_3d( 
    df, 
    x="x", 
    y="y", 
    z="z", 
    color="label" 
) 
fig.show()