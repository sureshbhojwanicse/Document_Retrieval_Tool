from dependency import *

class Joke(BaseModel):
    answer: str = Field(description="The answer that solves the user queries based on the question and context provided")
    reason: str = Field(description="Provide a valid reason why you provided this answer for the user")
    relevancy: str =Field(description="Provide how much relevant the context is based on the question and the answer that you have provided")


class ALIM():
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('cl100k_base')

    
    def tiktoken_len(self,text):

        tokens = self.tokenizer.encode(

            text,

            disallowed_special=()

        )

        return len(tokens)




    def embeddings_creation(self,data,client,filename,collection_name,flag=0):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=self.tiktoken_len,
        separators=["\n\n", "\n", " "]
        )
        data=str(data)
        if flag==1:
            chunks=text_splitter.split_text(data)
        else:
            chunks=[data]
        # print(chunks)
        # chunks=data.split("page_splitter")

        title = f"Document of pharmaceutical- {client} and Drug - {collection_name}"
        sk=[]
        for embed in tqdm(chunks):
            model="text-embedding-ada-002"
            text = embed.replace("\n", " ")
            sk.append(client_openai.embeddings.create(input = [text], model=model).data[0].embedding)

        json_file={f"{collection_name}":chunks,"embeddings":sk}
        out_file = open(f"data_client/data_client_embedding/{collection_name}.json", "w")  
        json.dump(json_file, out_file)  
        out_file.close()
    
        print("done")

    def llm_openai(self, query, context):
    # Joke=Joke()
        parser = PydanticOutputParser(pydantic_object=Joke)
        prompt = PromptTemplate(
        template="""
        You are an expert Islamic Chatbot tasked with answering any question about Islam.
        Generate a comprehensive and informative answer of 100 words or less for the \
        given question based solely on the provided information. You must \
        only use information from the provided information. Use an unbiased and \
        Islamic Scholar tone. Combine information provided together into a coherent answer. Do not \
        repeat text. If there is reference present in the information provided then provide a reference(Verse, Chapter number or Chapter name or Quran Verse) in the answer make sure to enclose it in sqare brackets([])\n
        CONTEXT:
            {content}
            Question:{query}
            {format_instructions}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions(),"content":context},
    )
        input= prompt.format_prompt(query=query).to_string()
        messages= [HumanMessage(content=input)]
        output=model(messages).content
        try:
            out_dict=parser.parse(output).dict()
        except:
            try:
                fix_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
                out_dict=fix_parser.parse(output).dict()
            except Exception as e:
                print("Error in the dictionary",e)
                out_dict="Sorry there was an issue with at the servers please refresh and try again"
        return out_dict
    
    def similarity_search_score(self, query= "", collection_name = '', vector = None ):
        index_key = None
        persistent_client = chromadb.PersistentClient()
        # collection = persistent_client.get_collection(collection_name)
        embeddings =  OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
        langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        )
        if vector:
            return collection_name, langchain_chroma.similarity_search_by_vector(vector, k = 3)
        return collection_name, langchain_chroma.similarity_search_with_score(query, k = 3)
    

    def get_context_page(self,question,collection_name):
        val=self.similarity_search_score(question,collection_name)
        map_context= {}
        for i, value in enumerate(val[1]):
            map_context[value[0].page_content]=value[1]

        val=sorted(map_context, key=map_context.get, reverse=True)[:4]

        val_map={}
        val_map[f"{collection_name}_A"] = val[0]
        val_map[f"{collection_name}_B"] = val[1]
        # val_map[f"{collection_name}_C"] = val[2]

        ##Combination of all the values
        context=""
        for con in list(val_map.keys()):
            context= context+ f"{val_map[con]}{con}\n" +"\n"

        return val_map, context