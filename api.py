import json, fitz, os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import create_model, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import List

load_dotenv()
app = FastAPI()

def build_schema(schema_dict: dict, name: str = "Ext"):
    fields = {}
    for k, v in schema_dict.items():
        if isinstance(v, dict):
            fields[k] = (build_schema(v, f"{name}_{k}"), Field(description=f"Ext {k}"))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            fields[k] = (List[build_schema(v[0], f"{name}_{k}_i")], Field(description=f"Ext list {k}"))
        else:
            fields[k] = ({"int":int, "float":float, "list":list}.get(str(v).lower(), str), Field(description=f"Ext {k}"))
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
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(build_schema(s_dict))
        return llm.invoke(f"Extract:\n{text}").model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


        

