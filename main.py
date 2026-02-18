import os
import json
import fitz
from pydantic import create_model, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv


def create_dynamic_schema(fields_config):  ##this function dynamically create pydantic model
    dynamic_fields = {}    ##create a empty dict to store model fields
    for field_name, field_info in fields_config.items():  ##loop through user-defined fields
        field_type = field_info.get('type', str)    #gets the type (defult = str if not given)
        description = field_info.get('description', '')   ##get sescriptions for thee fild
        dynamic_fields[field_name] = (field_type, Field(description=description))  ##create a pydantic fild dynamically
    
    return create_model('DynamicExtraction', **dynamic_fields)


def get_schema_from_user():
    print("\n=== Define Your Schema ===")
    fields_config = {}
    
    while True:
        field_name = input("Field name (or 'done'): ").strip()
        if field_name.lower() == 'done':
            break
        
        field_type_str = input("Type (str/int/float/list): ").strip().lower()
        type_map = {'str': str, 'int': int, 'float': float, 'list': list}
        field_type = type_map.get(field_type_str, str)
        
        description = input("Description: ").strip()
        fields_config[field_name] = {'type': field_type, 'description': description}
    
    return fields_config



def main():
    load_dotenv()
    
    fields_config = get_schema_from_user()
    if not fields_config:
        print("No fields defined.")
        return
    
    DynamicModel = create_dynamic_schema(fields_config)
    
    file_path = input("\nEnter PDF file path: ").strip().strip('\'"')
    if not os.path.exists(file_path):
        print("File not found.")
        return
    
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(DynamicModel)
    data = structured_llm.invoke(f"Extract data:\n\n{text}")
    
    output = json.dumps(data.model_dump(), indent=2)
    print(output)
    
    with open("latest_output.json", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()