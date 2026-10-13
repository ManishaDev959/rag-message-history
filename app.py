# Loader (reads web pages)
from langchain_community.document_loaders import WebBaseLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# Prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Output parser
from langchain_core.output_parsers import StrOutputParser

# Runnables
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Message history
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever

from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()

loader = WebBaseLoader('https://www.flipkart.com/google-pixel-11-obsidian-256-gb/p/itma0200211f30ae?pid=MOBHPHNV4FA6ZKGF&lid=LSTMOBHPHNV4FA6ZKGFNVP4HH&marketplace=FLIPKART&q=google+pixel+10&store=tyy%2F4io&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_12_na_na_ps&otracker1=AS_QueryStore_OrganicAutoSuggest_1_12_na_na_ps&fm=Search&iid=en_bOj8XiqebtpZTj28U-f9Ty29095STQ6XH67CZUjYi0L9vwZ154ODFnDR2WaU9fSw3Hv2p8BhS3_fK8i1eDcnxWxyvX28L9s7p0i8zKaVMsSRuhzvf51dw92r84Q7gLae&ppt=sp&ppn=sp&ssid=kwwim3dpw00000001791888954095&qH=3338507ebe32272d&ov_redirect=true')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap= 200)
chunks = text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents=chunks,
                                     embedding=embeddings, persist_directory="./chroma_db")


retriever = vectorstore.as_retriever()


system_data = """You are a helpful Question Answering System.
Now Answer the query from the provided context.
If you dont know the answer, directly say I dont know the answer.
Keep the answer concise. {context}
"""

prompt = ChatPromptTemplate.from_messages([('system', system_data),
                                  ('human', '{input}')])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# response = rag_chain.invoke({'input': 'What was my previous query?'})


# print(response)


context_system = """Given a chat history and the latest user question
which might reference the chat history, rewrite it as a standalone question.
Do NOT answer it, just rewrite it if needed, otherwise return it as is."""

# Rephrasing prompt: NO {context}
context_prompt = ChatPromptTemplate.from_messages([
    ("system", context_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Answering prompt: HAS {context}
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Question Answering System.
Answer the query from the provided context or the chat history.
If you dont know the answer, directly say I dont know the answer.
Keep the answer concise.

Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


history_aware_retreiver = create_history_aware_retriever(llm, retriever, context_prompt)

question_answer_chain_2 = create_stuff_documents_chain(llm, qa_prompt)

rag_chain_2 = create_retrieval_chain(history_aware_retreiver, question_answer_chain_2)

chat_history = []

question = "What is the price of Google Pixel 11?"


response2 = rag_chain_2.invoke({'input' : question, 'chat_history': chat_history})

chat_history.extend([HumanMessage(content = question ), AIMessage(content = response2['answer'])])

question_2 = "What was my previous question ?"

response_3 = rag_chain_2.invoke({'input' : question_2, 'chat_history': chat_history})

print(response_3['answer'])