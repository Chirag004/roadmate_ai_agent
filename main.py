from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model= "llama3.2")

template = '''
You are an expert in aswering the questions related to road travel, travel itinerary, travel , itinerary & travel rated small advices

here is the relevant data: {data}

here is the question to be answered: {question}
'''
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("\n\n--------------------------------------")
    question = input("Ask your question ( press q to quit) :- ")
    print("\n\n")
    if question == "q":
        print("----exit----")
        break

    data = retriever. invoke (question)
    result = chain.invoke({"data":data, "question": question})
    print(result)