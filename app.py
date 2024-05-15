from dependency import *
from main import *

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



@app.post("/islamic_encyclopedia",tags=["Question Answering"])
async def answerapi(request: Request):
    try:
        start_time=time.perf_counter
        data=await request.json()
        question=data["question"]
        alim=ALIM()
        val_map, context= alim.get_context_page(question, "Islam_Gordon")
        output_dict=alim.llm_openai(question,context)
        return {"Response":output_dict}


    except Exception as e:
        print("Exception at api",e)
