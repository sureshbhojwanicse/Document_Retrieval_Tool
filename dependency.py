from flask import Flask,request
import os
# import google.generativeai as genai
import json
import langchain
import openai
from datetime import datetime
import pandas as pd
from tqdm import tqdm
openai.api_key="sk-proj-9oFspACwXLo6tjgsobG7T3BlbkFJIXBM10NHcdXgdU5F9nQH"
openai.api_key="sk-proj-9oFspACwXLo6tjgsobG7T3BlbkFJIXBM10NHcdXgdU5F9nQH"
os.environ["OPENAI_API_KEY"] = "sk-proj-9oFspACwXLo6tjgsobG7T3BlbkFJIXBM10NHcdXgdU5F9nQH"
openai.api_key = "sk-proj-9oFspACwXLo6tjgsobG7T3BlbkFJIXBM10NHcdXgdU5F9nQH"
# from openai import OpenAI
# client=OpenAI()
import tiktoken
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain, TransformChain
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
)
from langchain.prompts import (
    PromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAI
from pydantic import BaseModel, Field
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
model = ChatOpenAI(temperature=0)
from openai import OpenAI
client_openai = OpenAI()
from IPython.display import Markdown
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from fastapi import FastAPI, Form, Request, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import warnings
import time
warnings.filterwarnings('ignore')


