from langchain_community.document_loaders import PyPDFLoader 
loader = PyPDFLoader('dl-curriculum.pdf') 
docs = loader.load() 
print(len(docs)) 
print(docs[0].page_content) 
print(docs[1].metadata)
from langchain_community.document_loaders import CSVLoader 
loader = CSVLoader(file_path='Social_Network_Ads.csv') 
docs = loader.load() 
print(len(docs)) 
print(docs[1])
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, 
WebBaseLoader, TextLoader 
loaders = { 
"PDF": PyPDFLoader("dl-curriculum.pdf"), 
"CSV": CSVLoader("Social_Network_Ads.csv"), 
"WEB": WebBaseLoader("https://www.langchain.com"), 
"TEXT": TextLoader("cricket.txt") 
} 
for name, loader in loaders.items(): 
docs = loader.load() 
print(f"\n===== {name} Document Loader =====") 
print("Sample text:", docs[0].page_content[:200]) 
print("Metadata:", docs[0].metadata)