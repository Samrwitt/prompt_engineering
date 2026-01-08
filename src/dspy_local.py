from __future__ import annotations
import dataclasses
from typing import Dict, List, Type, Any, Optional

# --- Fields ---

@dataclasses.dataclass
class Field:
    """
    Represents an Input or Output field in a Signature.
    """
    desc: str
    prefix: str = ""

class InputField(Field):
    pass

class OutputField(Field):
    pass

# --- Signature ---

class SignatureMeta(type):
    def __new__(cls, name, bases, namespace):
        # Extract fields from type annotations
        annotations = namespace.get('__annotations__', {})
        new_namespace = {'_inputs': {}, '_outputs': {}, **namespace}
        
        for field_name, field_type in annotations.items():
            # In a real DSPy, we'd check if field_type is InputField or OutputField
            # Here we assume user defines class attributes for the Field specs
            pass
            
        return super().__new__(cls, name, bases, new_namespace)

class Signature:
    """
    Base class for DSPy-like Signatures.
    Users should define the docstring and class attributes for fields.
    """
    __doc__: str = ""
    
    def __init__(self):
        pass

    @classmethod
    def instructions(cls) -> str:
        return cls.__doc__.strip()

# --- Modules ---

class Module:
    def __init__(self):
        pass
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Predict(Module):
    def __init__(self, signature: Type[Signature]):
        self.signature = signature

class ChainOfThought(Module):
    def __init__(self, signature: Type[Signature]):
        self.signature = signature

# --- Compiler ---

def compile_to_vector(module: Module, blocks: List[str], verbose: bool = False) -> List[int]:
    """
    "Compiles" a DSPy module (Predict/CoT + Signature) into a 
    binary selection mask over the available prompt blocks.
    
    This simulates DSPy's 'Teleprompter' selection but in a fixed rule-based way
    to serve as a strong baseline.
    """
    x = [0] * len(blocks)
    sig = module.signature
    
    # 1. Introspect constraints from Signature
    instructions = sig.instructions().lower()
    
    # Determine constraints based on signature class name or docstring
    needs_numbers = "number" in instructions or "integer" in instructions or "calc" in instructions
    needs_yesno = "yes" in instructions or "no" in instructions
    
    # 2. Introspect Strategy from Module Type
    is_cot = isinstance(module, ChainOfThought)
    
    for i, b in enumerate(blocks):
        text = b.lower()
        
        # --- Logic mirroring dspylike_baseline_vector but driven by the Module ---
        
        # A. Reliability (Universal to DSPy predictors)
        if any(w in text for w in ["guess", "unsure", "estimate"]):
            continue

        # B. Reasoning (Dependent on Module Type)
        is_cot_block = any(w in text for w in ["step by step", "break down", "thinking"])
        if is_cot:
            if is_cot_block:
                x[i] = 1 # ACTIVATED CoT
        else:
            if is_cot_block:
                continue # BLOCKED CoT for standard Predict
        
        # C. Task/Format Constraints (Dependent on Signature)
        # 1. Format: Integer
        if needs_numbers and any(w in text for w in ["integer", "number", "math"]):
            x[i] = 1
            
        # 2. Format: Yes/No
        if needs_yesno and any(w in text for w in ["yes", "no"]):
            x[i] = 1
            
        # 3. Instruction: Solve/Correctly (General)
        if "solve" in text or "correctly" in text or "answer" in text:
            # General instructions are usually good unless they contradict
            x[i] = 1
            
        # 4. Negative Constraints (do not explain)
        # Often useful for "Predict" (Direct) but maybe less for CoT?
        # A standard predictor usually wants just the answer.
        if "explanation" in text:
             if "no" in text or "do not" in text:
                 # "Do not explain"
                 if not is_cot: 
                     x[i] = 1
                 else:
                     # If CoT, we DO want explanation usually, but often prompt says 
                     # "Think step by step... output final answer". 
                     # Let's keep "no explanation" off for CoT usually, or be careful.
                     # For this baseline, let's say CoT implies we want reasoning trace, 
                     # but maybe the final answer block says "output only answer"?
                     # It's safer to avoid "do not explain" if we encourage CoT.
                     pass 

    # Fallback
    if sum(x) == 0 and blocks:
        x[0] = 1
        
    return x
