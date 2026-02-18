import os
import json
import fitz
from pydantic import create_model, Field, ValidationError
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pathlib import Path


def create_dynamic_schema(fields_config):
    """
    Dynamically create a Pydantic model based on user-defined field configuration.
    
    Args:
        fields_config (dict): Dictionary containing field names and their configurations
                            Example: {'name': {'type': str, 'description': 'Person name'}}
    
    Returns:
        type: Dynamically created Pydantic model class
    """
    dynamic_fields = {}
    for field_name, field_info in fields_config.items():
        field_type = field_info.get('type', str)
        description = field_info.get('description', '')
        dynamic_fields[field_name] = (field_type, Field(description=description))
    
    return create_model('DynamicExtraction', **dynamic_fields)


def get_schema_from_user():
    """
    Interactive function to get field schema from user input.
    Allows user to define custom fields with types and descriptions.
    
    Returns:
        dict: Configuration dictionary with field definitions
    """
    print("\n=== Define Your Schema ===")
    print("Define the fields you want to extract from the PDF.")
    print("Supported types: str, int, float, list\n")
    
    fields_config = {}
    
    while True:
        field_name = input("Field name (or 'done' to finish): ").strip()
        if field_name.lower() == 'done':
            break
        
        if not field_name:
            print("Field name cannot be empty. Try again.")
            continue
        
        field_type_str = input("Type (str/int/float/list): ").strip().lower()
        type_map = {'str': str, 'int': int, 'float': float, 'list': list}
        
        if field_type_str not in type_map:
            print(f"Invalid type. Using 'str' as default.")
            field_type = str
        else:
            field_type = type_map[field_type_str]
        
        description = input("Description: ").strip()
        fields_config[field_name] = {'type': field_type, 'description': description}
        print(f"✓ Added field '{field_name}' ({field_type_str})\n")
    
    return fields_config



def extract_text_from_pdf(file_path):
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text from all pages
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    try:
        doc = fitz.open(file_path)
        text = ""
        total_pages = doc.page_count
        
        print(f"Extracting text from {total_pages} page(s)...")
        for page_num, page in enumerate(doc, 1):
            text += page.get_text()
            print(f"  ✓ Processed page {page_num}/{total_pages}")
        
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def extract_json_from_text(text, dynamic_model, api_key=None):
    """
    Use Groq LLM to extract structured data from text.
    
    Args:
        text (str): Text to extract data from
        dynamic_model (type): Pydantic model for structured output
        api_key (str, optional): Groq API key. If None, uses GROQ_API_KEY env var
    
    Returns:
        dict: Extracted structured data
    
    Raises:
        Exception: If LLM call fails
    """
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        structured_llm = llm.with_structured_output(dynamic_model)
        
        print("\nExtracting structured data using Groq LLM...")
        data = structured_llm.invoke(f"Extract data from the following text:\n\n{text}")
        
        return data.model_dump()
    except ValidationError as e:
        raise ValueError(f"Data validation error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error during LLM extraction: {str(e)}")


def save_output(data, output_file="latest_output.json"):
    """
    Save extracted JSON data to file.
    
    Args:
        data (dict): Data to save
        output_file (str): Output file path
    
    Returns:
        str: Formatted JSON string
    """
    output = json.dumps(data, indent=2)
    
    try:
        output_path = Path(output_file)
        output_path.write_text(output)
        print(f"\n✓ Output saved to: {output_path.absolute()}")
    except Exception as e:
        print(f"⚠ Warning: Could not save to file: {str(e)}")
    
    return output


def main():
    """Main execution function."""
    load_dotenv()
    
    try:
        print("=" * 50)
        print("PDF to JSON Extractor")
        print("=" * 50)
        
        # Get schema from user
        fields_config = get_schema_from_user()
        if not fields_config:
            print("\n⚠ No fields defined. Exiting.")
            return
        
        # Create dynamic Pydantic model
        DynamicModel = create_dynamic_schema(fields_config)
        print(f"\n✓ Schema created with {len(fields_config)} field(s)")
        
        # Get PDF file path
        file_path = input("\nEnter PDF file path: ").strip().strip('\'"')
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        print(f"\n✓ Extracted {len(text)} characters from PDF")
        
        # Extract structured data using LLM
        data = extract_json_from_text(text, DynamicModel)
        
        # Save and display output
        output = save_output(data)
        
        print("\n" + "=" * 50)
        print("Extracted Data (JSON):")
        print("=" * 50)
        print(output)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
    except ValueError as e:
        print(f"\n✗ Configuration Error: {str(e)}")
    except Exception as e:
        print(f"\n✗ Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()