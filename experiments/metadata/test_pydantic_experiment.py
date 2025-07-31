"""
Experiment to test if Pydantic can work and how it compares to dataclasses.
"""

import json

# First, let's check if Pydantic is available
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
    print("✅ Pydantic is available!")
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("❌ Pydantic is not installed")
    print("To install: pip install pydantic")

# Let's also test with dataclasses for comparison
from dataclasses import asdict, dataclass


# --- DATACLASS APPROACH (Current) ---
@dataclass
class DataclassExportContext:
    """Export context using dataclasses."""
    timestamp: str = "2024-01-01T00:00:00Z"
    strategy: str = "htp"
    version: str = "1.0"
    
    
@dataclass 
class DataclassModelInfo:
    """Model info using dataclasses."""
    name_or_path: str
    class_name: str  # Note: can't use 'class' as field name
    total_modules: int = 0


# --- PYDANTIC APPROACH (Proposed) ---
if PYDANTIC_AVAILABLE:
    class PydanticExportContext(BaseModel):
        """Export context using Pydantic."""
        timestamp: str = Field(default="2024-01-01T00:00:00Z", description="Export timestamp")
        strategy: str = Field(default="htp", description="Export strategy")
        version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
        
    class PydanticModelInfo(BaseModel):
        """Model info using Pydantic."""
        name_or_path: str = Field(description="Model name or path")
        class_: str = Field(alias="class", description="Model class name")  # Can use alias!
        total_modules: int = Field(default=0, ge=0)  # Can add constraints!


def experiment():
    """Run experiments to compare approaches."""
    
    print("\n=== DATACLASS APPROACH ===")
    # Create with dataclass
    dc_context = DataclassExportContext()
    dc_model = DataclassModelInfo(name_or_path="bert-base", class_name="BertModel")
    
    # Convert to dict
    dc_dict = {
        "export_context": asdict(dc_context),
        "model": {
            "name_or_path": dc_model.name_or_path,
            "class": dc_model.class_name,  # Manual rename needed
            "total_modules": dc_model.total_modules
        }
    }
    
    print("Dataclass result:")
    print(json.dumps(dc_dict, indent=2))
    
    # No built-in validation
    print("\nDataclass validation: None built-in")
    print("Schema generation: Manual only")
    
    if PYDANTIC_AVAILABLE:
        print("\n=== PYDANTIC APPROACH ===")
        # Create with Pydantic
        pd_context = PydanticExportContext()
        pd_model = PydanticModelInfo(name_or_path="bert-base", **{"class": "BertModel"})
        
        # Convert to dict (automatic aliasing!)
        pd_dict = {
            "export_context": pd_context.model_dump(),
            "model": pd_model.model_dump(by_alias=True)
        }
        
        print("Pydantic result:")
        print(json.dumps(pd_dict, indent=2))
        
        # Automatic validation
        print("\nPydantic validation: Built-in")
        try:
            # This will fail validation
            bad_context = PydanticExportContext(version="1.2.3")  # Pattern mismatch
        except Exception as e:
            print(f"Validation error caught: {e}")
            
        # Automatic schema generation
        print("\nPydantic JSON Schema:")
        schema = PydanticModelInfo.model_json_schema()
        print(json.dumps(schema, indent=2)[:200] + "...")
        
        print("\n=== BENEFITS OF PYDANTIC ===")
        print("1. ✅ Automatic validation with constraints")
        print("2. ✅ JSON Schema generation")
        print("3. ✅ Field aliases (can use 'class' properly)")
        print("4. ✅ Better error messages")
        print("5. ✅ Type coercion")
        print("6. ✅ Serialization control")
        
    print("\n=== MIGRATION PATH ===")
    print("1. Check if Pydantic is in dependencies")
    print("2. If not, add to pyproject.toml")
    print("3. Replace dataclasses gradually")
    print("4. Use Pydantic's schema generation for validation")


if __name__ == "__main__":
    experiment()