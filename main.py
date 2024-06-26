from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import sys

__author__ = "Pranav Gupta"

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

loader = TextLoader("flightplanweights.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # search_type="similarity", 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


examples = [

        {
        "question": "I want to travel the shortest distance at the lowest altitude",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: For every altitude, which is the shortest flight plan?
        Intermediate Answer: 
        'flight_plan_1': [12.1, 1000, 1108.57, 10],
        'flight_plan_5': [15.1, 1500, 1148.77, 9],
        'flight_plan_9': [17.1, 2500, 1236.27, 10],
        'flight_plan_13': [19.1, 3000, 1112.14, 9],
        Follow up: Which one is at the lowest altitude?
        Intermediate Answer:
        'flight_plan_1': [12.1, 1000, 1108.57, 10]
        So the final answer is FLIGHT PLAN: flight_plan_1.
        """,
    },
     {
        "question": "I want to use the least amount of energy while traveling the longest distance",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What are the 5 flight plans with the lowest energy consumption
        Intermediate Answer: 
        'flight_plan_7': [15.7, 1500, 1084.43, 11],
        'flight_plan_8': [15.9, 1500, 1091.84, 14],
        'flight_plan_6': [15.4, 1500, 1095.56, 10],
        'flight_plan_16': [19.9, 3000, 1101.97, 14]
        'flight_plan_15': [19.7, 3000, 1108.04, 12],
        Follow up: Which one has the longest distance?
        Intermediate Answer:
        'flight_plan_16': [19.9, 3000, 1101.97, 14]
        So the final answer is FLIGHT PLAN: flight_plan_16.
        """,
    },
    {
        "question": "I want to take the least complex path that covers the longest distance",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What does complex mean?
        Intermediate Answer: complex means waypoints.
        Follow up: What are the flight paths with the lowest number of waypoints
        Intermediate Answer: 
        'flight_plan_5': [15.1, 1500, 1148.77, 9],
        'flight_plan_13': [19.1, 3000, 1112.14, 9],
        Follow up: Which one has the longest distance?
        Intermediate Answer:
        'flight_plan_13': [19.1, 3000, 1112.14, 9],
        So the final answer is FLIGHT PLAN: flight_plan_13.
        """,
    },
    {
        "question": "I want to take a moderately complex path that is at the highest altitude and uses the least energy",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What does complex mean?
        Intermediate Answer: complex means waypoints.
        Follow up: What does moderately complex mean?
        Intermediate Answer: moderately complex means 11-13 waypoints.
        Follow up: What are the flight paths with 11-13 waypoints.
        Intermediate Answer: 
        'flight_plan_2': [12.4, 1000, 1230.18, 11],
        'flight_plan_3': [12.7, 1000, 1233.11, 13],
        'flight_plan_7': [15.7, 1500, 1084.43, 11],
        'flight_plan_10': [17.4, 2500, 1227.38, 11],
        'flight_plan_11': [17.7, 2500, 1223.45, 13],
        'flight_plan_15': [19.7, 3000, 1108.04, 12],
        Follow up: Which ones are at the highest altitudes?
        Intermediate Answer:
        'flight_plan_10': [17.4, 2500, 1227.38, 11],
        'flight_plan_11': [17.7, 2500, 1223.45, 13],
        'flight_plan_15': [19.7, 3000, 1108.04, 12],
        Follow up: Which one uses the least energy?
        Intermediate Answer: 'flight_plan_15': [19.7, 3000, 1108.04, 12],
        So the final answer is FLIGHT PLAN: flight_plan_15.
        """,
    },

]


weightedexamples = [

        {
        "question": "I want to travel the shortest distance at the lowest altitude",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What categories does the user seem to care about?
        Intermediate Answer: 
        Distance and Altitude
        Follow up: Did they place importance on one or do they care about both equally?
        Intermediate Answer:
        Both equally
        So the final answer is [0.5,0.5,0,0]
        """,
    },
    {
        "question": "I am low on fuel",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What categories does the user seem to care about?
        Intermediate Answer: 
        Energy Consumption
        Follow up: Did they place importance on one or do they care about both equally?
        Intermediate Answer:
        They only care about one thing
        So the final answer is [0,0,1,0, because the only important thing here is fuel.]
        """,
    },
     {
        "question": "I want to use the least amount of energy while traveling the longest distance",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What categories does the user seem to care about?
        Intermediate Answer: 
        Energy Consumption and Distance
        Follow up: Did they place importance on one or do they care about both equally?
        Intermediate Answer:
        both equally
        So the final answer is [0.5, 0, 0,5, 0, I detected you care equally about distance and energy, so they are weighted equally.]
        """,
    },
    {
        "question": "I hate complicated paths and I want to travel at a low altitude",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What does complicated or complex mean?
        Intermediate Answer: complicated or compelx means waypoints.
        Follow up: What categories does the user seem to care about?
        Intermediate Answer: 
        Waypoints and altitude
        Follow up: Did they place importance on one or do they care about both equally?
        Intermediate Answer:
        They said they hate complicated paths, so that is more important
        So the final answer is [0, 0.3, 0, 0.7, I detected you find complicated paths more important than altitude. ]
        """,
    },
    {
        "question": "I want to take a moderately complex path that is at the highest altitude and really uses the least energy.",
        "context": "{context}",
        "answer": """
        Are follow up questions needed here: Yes.
        Follow up: What does complex mean?
        Intermediate Answer: complex means waypoints.
        Follow up: What categories does the user seem to care about?
        Intermediate Answer:
        Waypoints, Altitude, Energy Consumption
        Follow up: Did they place importance on one or do they care about them equally?
        Intermediate Answer: 
        The use of really implies they care about energy consumption slightly more than waypoints and altitude
        So the final answer is [0,0.3,0.4,0.3, I detected you care about energy slightly more than complexity and altitude.]
        """,
    },

]

example_prompt = PromptTemplate(
    input_variables=["question", "context", "answer"], template="Question: {question}\n{answer}"
)

fsprompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {question} {context}",
    input_variables=["question", "context"],
)

weighted_fsprompt = FewShotPromptTemplate(
    examples=weightedexamples,
    example_prompt=example_prompt,
    suffix="Question: {question} {context}",
    input_variables=["question", "context"],
)

prompt = hub.pull("rlm/rag-prompt")


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | weighted_fsprompt
    | llm
    | StrOutputParser()
    )

while True:
    query = input("Question: ")

    if query in ['quit', 'q', 'exit']:
        sys.exit()
    
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    
    print('\n')

# rag_chain.invoke("We want to use the least amount of fuel for our flight and we don't care about anything else, also give me the weights you assigned to each category")