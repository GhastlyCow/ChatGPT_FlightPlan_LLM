from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

flight_plans = {

    # You are an expert on determining the best flight plan. Given an input question, pick the best flight plan that satisifies the users constraints in a friendly way.
    # The elements in the list are:
    # D - Distance
    # A - Altitude
    # E - Energy Consumption
    # W - Number of Waypoints
    'flight_plan_1': [12.4, 1000, 1108.57, 10],
    'flight_plan_2': [12.4, 1000, 1230.18, 11],
    'flight_plan_3': [12.4, 1000, 1233.11, 13],
    'flight_plan_4': [12.4, 1000, 1221.78, 15],

    'flight_plan_5': [12.4, 1500, 1148.77, 9],
    'flight_plan_6': [12.4, 1500, 1095.56, 10],
    'flight_plan_7': [12.4, 1500, 1084.43, 11],
    'flight_plan_8': [12.4, 1500, 1091.84, 14],

    'flight_plan_9': [12.4, 2500, 1236.27, 10],
    'flight_plan_10': [12.4, 2500, 1227.38, 11],
    'flight_plan_11': [12.4, 2500, 1223.45, 13],
    'flight_plan_12': [12.4, 2500, 1218.78, 15],

    'flight_plan_13': [12.4, 3000, 1112.14, 9],
    'flight_plan_14': [12.4, 3000, 1108.57, 10],
    'flight_plan_15': [12.4, 3000, 1108.04, 12],
    'flight_plan_16': [12.4, 3000, 1101.97, 14]

}

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

loader = TextLoader("flightplans.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # search_type="similarity", 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""
# custom_rag_prompt = PromptTemplate.from_tempate(template)
prompt = hub.pull("rlm/rag-prompt")


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("We want to use the least amount of fuel for our flight, also give me the weights you assigned to each category"):
    print(chunk, end="", flush=True)
# rag_chain.invoke("Which flight should i choose if I want to be at 1500 altitude and use the least amount of energy?")