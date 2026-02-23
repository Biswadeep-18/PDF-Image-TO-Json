import json, fitz, os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import create_model, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import List, Optional, Any

load_dotenv()
app = FastAPI()

def build_schema(schema_dict: dict, name: str = "Ext"):
    fields = {}
    for k, v in schema_dict.items():
        if isinstance(v, dict):
            fields[k] = (Optional[build_schema(v, f"{name}_{k}")], Field(default=None, description=f"Ext {k}"))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            fields[k] = (Optional[List[build_schema(v[0], f"{name}_{k}_i")]], Field(default=None, description=f"Ext list {k}"))
        else:
            base_type = {"int": int, "float": float, "list": list}.get(str(v).lower(), str)
            fields[k] = (Optional[base_type], Field(default=None, description=f"Ext {k}"))
    return create_model(name, **fields)

@app.get("/")
def root(): return {"status": "running"}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    schema: str = Form('{"vendor": "str", "items": [{"name": "str", "price": "float"}]}')
):
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "Only PDF supported")
    try:
        s_dict = json.loads(schema.replace("'", '"'))
    except:
        raise HTTPException(400, f"Invalid JSON: {schema}")

    text = "".join(p.get_text() for p in fitz.open(stream=await file.read(), filetype="pdf"))
    if not text.strip(): raise HTTPException(422, "No text found")

    try:
        print(text)
        print(s_dict)
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(build_schema(s_dict))
        prompt = (
            f"SYSTEM: You are a strict data extractor. ONLY extract information that is explicitly present in the provided text. "
            f"If a piece of information is not found in the text, DO NOT make it up. DO NOT hallucinate. "
            f"If the schema requires a field that is missing from the text, use null or an empty value as appropriate for the type. "
            f"Strictly follow the source text.\n\n"
            f"Source Text:\n{text}\n\n"
            f"Extract the following data based on the source text above."
        )
        return llm.invoke(prompt).model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


        

