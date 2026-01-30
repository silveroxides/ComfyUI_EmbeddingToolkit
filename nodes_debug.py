import torch
import os
import folder_paths
import comfy.utils
import re

class PromptEmbeddingFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fix_prompt"
    CATEGORY = "EmbeddingToolkit/Utils"

    def fix_prompt(self, text):
        # Add space before embedding if missing
        text = re.sub(r'(?<!\s)(embedding:[^\s]+)', r' \1', text)
        # Add space after embedding if missing
        text = re.sub(r'(embedding:[^\s]+)(?!\s)', r'\1 ', text)
        return (text,)

class InspectEmbeddingForClip:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            embedding_files = [f for f in folder_paths.get_filename_list("embeddings") if f.lower().endswith((".safetensors", ".pt"))]
        except:
            embedding_files = ["None"]
            
        return {
            "required": {
                "clip": ("CLIP",),
                "embedding_file": (embedding_files,),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect_embedding"
    CATEGORY = "EmbeddingToolkit/Debug"
    OUTPUT_NODE = True

    def inspect_embedding(self, clip, embedding_file):
        output_text = []
        output_text.append(f"Inspecting File: {embedding_file}")
        
        # Load embedding
        full_path = folder_paths.get_full_path("embeddings", embedding_file)
        if not full_path:
             output_text.append("Error: Embedding file not found.")
             return {"ui": {"text": ["\n".join(output_text)]}}
             
        try:
            embed_data = comfy.utils.load_torch_file(full_path)
            output_text.append(f"Keys found in file: {list(embed_data.keys())}")
        except Exception as e:
            output_text.append(f"Error loading file: {e}")
            return {"ui": {"text": ["\n".join(output_text)]}}
            
        # Inspect CLIP Wrapper
        cond_model = clip.cond_stage_model
        output_text.append(f"\nCLIP Model Wrapper: {type(cond_model).__name__}")
        
        # Inspect Tokenizer Wrapper
        tokenizer = clip.tokenizer
        output_text.append(f"Tokenizer Wrapper: {type(tokenizer).__name__}")
        if hasattr(tokenizer, "llama_template"):
             output_text.append(f"  Template: {repr(tokenizer.llama_template)}")
        
        # Check sub-models and sub-tokenizers
        potential_parts = [
            'clip_l', 'clip_g', 'clip_h', 
            't5xxl', 't5xl', 't5base',
            'qwen25_7b', 'qwen25_3b', 'qwen3_4b', 'qwen3_8b', 'qwen3_2b', 'qwen3_06b',
            'mistral3_24b', 'gemma2_2b', 'gemma3_4b', 'gemma3_12b',
            'jina_clip_2', 'gemma', 'jina', 'byt5_small', 'llama'
        ]
        
        # Add dynamic parts from tokenizer
        if hasattr(tokenizer, "clip") and isinstance(tokenizer.clip, str):
            if tokenizer.clip not in potential_parts:
                potential_parts.append(tokenizer.clip)

        for part in potential_parts:
            # Check Model
            if hasattr(cond_model, part):
                sub_model = getattr(cond_model, part)
                output_text.append(f"\n--- Component: {part} ---")
                
                # Check Tokenizer for this component
                if hasattr(tokenizer, part):
                    sub_tok = getattr(tokenizer, part)
                    output_text.append(f"  Tokenizer: Found")
                    output_text.append(f"    Embedding Dir: {getattr(sub_tok, 'embedding_directory', 'N/A')}")
                    output_text.append(f"    Embedding ID: {repr(getattr(sub_tok, 'embedding_identifier', 'N/A'))}")
                else:
                    output_text.append(f"  Tokenizer: NOT FOUND on wrapper")

                # Check Model Embedding Size
                if hasattr(sub_model, "transformer") and sub_model.transformer is not None:
                    try:
                        weight = sub_model.transformer.get_input_embeddings().weight
                        expected_size = weight.shape[-1]
                        output_text.append(f"  Model Embedding Size: {expected_size}")
                        
                        # Check compatibility with file
                        match_found = False
                        
                        # 1. Check exact key match
                        if part in embed_data:
                            emb = embed_data[part]
                            output_text.append(f"  [MATCH] Key '{part}' found in file.")
                            output_text.append(f"    File Shape: {emb.shape}")
                            if emb.shape[-1] != expected_size:
                                output_text.append(f"    [FAIL] Dimension mismatch! {emb.shape[-1]} vs {expected_size}")
                            else:
                                output_text.append(f"    [OK] Dimensions match.")
                            match_found = True
                            
                        # 2. Check for single tensor fallback
                        if not match_found and len(embed_data) == 1:
                             val = list(embed_data.values())[0]
                             if hasattr(val, "shape") and val.ndim >= 2:
                                 output_text.append(f"  [CHECK] Checking single tensor fallback...")
                                 if val.shape[-1] == expected_size:
                                     output_text.append(f"    [OK] Fallback match! Shape: {val.shape}")
                                     match_found = True
                                 else:
                                     output_text.append(f"    [FAIL] Fallback mismatch. {val.shape[-1]} vs {expected_size}")
                        
                        if not match_found:
                             output_text.append(f"  [INFO] No compatible embedding found for this component.")
                             
                    except Exception as e:
                        output_text.append(f"  Error inspecting model component: {e}")
                else:
                    output_text.append("  (No transformer found)")

        return {"ui": {"text": ["\n".join(output_text)]}}

NODE_CLASS_MAPPINGS = {
    "PromptEmbeddingFixer": PromptEmbeddingFixer,
    "InspectEmbeddingForClip": InspectEmbeddingForClip,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptEmbeddingFixer": "Prompt Embedding Fixer",
    "InspectEmbeddingForClip": "Inspect Embedding For Clip",
}
