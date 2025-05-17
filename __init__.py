import torch
import os
import folder_paths
import comfy.utils

class SaveTokenEmbeddings:
    def __init__(self):
        self.output_dir_list = folder_paths.get_folder_paths("embeddings")
        if not self.output_dir_list:
            print("SaveTokenEmbeddings: Warning: 'embeddings' folder type not found. Falling back to main output directory.")
            self.output_dir_list = [folder_paths.get_output_directory()]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "filename_prefix": ("STRING", {"default": "token_embeds"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_token_embeddings"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit"

    def save_token_embeddings(self, clip, text, filename_prefix):
        if clip is None:
            raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveTokenEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        tokenized_text_with_weights = clip.tokenize(text)
        actual_clip_model_wrapper = clip.cond_stage_model
        
        sd_clip_instances = {}
        potential_clip_parts = {'l': 'clip_l', 'g': 'clip_g', 't5xxl': 't5xxl'}

        for key_suffix, attr_name in potential_clip_parts.items():
            if hasattr(actual_clip_model_wrapper, attr_name):
                model_instance = getattr(actual_clip_model_wrapper, attr_name)
                if model_instance is not None and hasattr(model_instance, "process_tokens") and hasattr(model_instance, "transformer"):
                    sd_clip_instances[key_suffix] = model_instance
                else:
                    if model_instance is None:
                        print(f"SaveTokenEmbeddings: Warning: Attribute '{attr_name}' is None. Skipping '{key_suffix}'.")
                    else:
                        print(f"SaveTokenEmbeddings: Warning: Attribute '{attr_name}' (type {type(model_instance)}) is not a valid SDClipModel. Skipping '{key_suffix}'.")
        
        if not sd_clip_instances and hasattr(actual_clip_model_wrapper, "process_tokens") and hasattr(actual_clip_model_wrapper, "transformer"):
            print(f"SaveTokenEmbeddings: Warning: No standard attributes. Fallback: cond_stage_model as 'l'.")
            sd_clip_instances['l'] = actual_clip_model_wrapper
        
        if not sd_clip_instances:
            print(f"SaveTokenEmbeddings: Error: No valid SDClipModel instances found.")
            return {"ui": {"text": ["Error: No valid SDClipModel instances found."]}}

        tensors_to_save = {}
        for key_suffix, sd_clip_model_inst in sd_clip_instances.items():
            current_token_segments_with_weights = None
            if isinstance(tokenized_text_with_weights, dict):
                current_token_segments_with_weights = tokenized_text_with_weights.get(key_suffix)
            elif key_suffix == 'l' and not isinstance(tokenized_text_with_weights, dict):
                current_token_segments_with_weights = tokenized_text_with_weights
            
            if not current_token_segments_with_weights:
                print(f"SaveTokenEmbeddings: Warning: No token data for '{key_suffix}'. Skipping.")
                continue

            if not hasattr(sd_clip_model_inst, 'transformer') or sd_clip_model_inst.transformer is None:
                 print(f"SaveTokenEmbeddings: Error: SDClipModel for '{key_suffix}' has invalid 'transformer'. Skipping.")
                 continue
            device = sd_clip_model_inst.transformer.get_input_embeddings().weight.device
            
            tokens_for_process_tokens_method = []
            for segment_with_weights in current_token_segments_with_weights:
                tokens_for_process_tokens_method.append([item[0] for item in segment_with_weights])
            
            if not tokens_for_process_tokens_method:
                print(f"SaveTokenEmbeddings: Warning: No token ID segments for '{key_suffix}'. Skipping.")
                continue

            embeds_out_full, attention_mask_full, _ = sd_clip_model_inst.process_tokens(
                tokens_for_process_tokens_method, device
            )
            
            all_actual_embeddings_for_this_clip_part = []
            num_segments = embeds_out_full.shape[0]
            for s in range(num_segments):
                current_segment_embeds = embeds_out_full[s]
                current_segment_mask = attention_mask_full[s]
                actual_embeddings_this_segment = current_segment_embeds[current_segment_mask == 1]
                if actual_embeddings_this_segment.shape[0] > 0:
                    all_actual_embeddings_for_this_clip_part.append(actual_embeddings_this_segment)
            
            if not all_actual_embeddings_for_this_clip_part:
                print(f"SaveTokenEmbeddings: Warning: No actual tokens for '{key_suffix}'. Skipping.")
                continue

            final_embeddings_for_this_clip_part = torch.cat(all_actual_embeddings_for_this_clip_part, dim=0)
            
            # --- CORRECTED KEY NAMING ---
            if key_suffix == 't5xxl':
                save_key_in_file = key_suffix  # Use 't5xxl' directly
            else:
                save_key_in_file = f"clip_{key_suffix}" # Use 'clip_l', 'clip_g'
            # --- END CORRECTION ---

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveTokenEmbeddings: Processed unweighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveTokenEmbeddings: Error: No unweighted embeddings generated.")
            return {"ui": {"text": ["Error: No unweighted embeddings generated."]}}

        subfolder_in_prefix, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0] 
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
        if subfolder_in_prefix:
            os.makedirs(full_output_folder, exist_ok=True)

        metadata = {}
        counter = 1
        max_counter = 99999
        while True:
            save_filename = f"{filename_base}_{counter:05}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)
            if not os.path.exists(save_path):
                break
            counter += 1
            if counter > max_counter:
                print(f"SaveTokenEmbeddings: Warning: Max file attempts for {filename_base}.")
                return {"ui": {"text": [f"Error: Max files for {filename_base}"]}}
        
        print(f"SaveTokenEmbeddings: Saving unweighted to: {save_path}")
        comfy.utils.save_torch_file(tensors_to_save, save_path, metadata=metadata)
        return {"ui": {"text": [f"Saved unweighted to {save_path}"]}}


class SaveWeightedEmbeddings:
    def __init__(self):
        self.output_dir_list = folder_paths.get_folder_paths("embeddings")
        if not self.output_dir_list:
            print("SaveWeightedEmbeddings: Warning: 'embeddings' folder type not found. Falling back to main output directory.")
            self.output_dir_list = [folder_paths.get_output_directory()]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "filename_prefix": ("STRING", {"default": "weighted_embed"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_weighted_embeddings"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit"

    def save_weighted_embeddings(self, clip, text, filename_prefix):
        if clip is None:
            raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveWeightedEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        tokenized_text_with_weights = clip.tokenize(text)
        actual_clip_model_wrapper = clip.cond_stage_model
        
        sd_clip_instances = {}
        potential_clip_parts = {'l': 'clip_l', 'g': 'clip_g', 't5xxl': 't5xxl'}

        for key_suffix, attr_name in potential_clip_parts.items():
            if hasattr(actual_clip_model_wrapper, attr_name):
                model_instance = getattr(actual_clip_model_wrapper, attr_name)
                if model_instance is not None and hasattr(model_instance, "process_tokens") and hasattr(model_instance, "transformer"):
                    sd_clip_instances[key_suffix] = model_instance
                else:
                    if model_instance is None:
                        print(f"SaveWeightedEmbeddings: Warning: Attribute '{attr_name}' is None. Skipping '{key_suffix}'.")
                    else:
                        print(f"SaveWeightedEmbeddings: Warning: Attribute '{attr_name}' (type {type(model_instance)}) is not a valid SDClipModel. Skipping '{key_suffix}'.")
        
        if not sd_clip_instances and hasattr(actual_clip_model_wrapper, "process_tokens") and hasattr(actual_clip_model_wrapper, "transformer"):
            print(f"SaveWeightedEmbeddings: Warning: No standard attributes. Fallback: cond_stage_model as 'l'.")
            sd_clip_instances['l'] = actual_clip_model_wrapper
        
        if not sd_clip_instances:
            print(f"SaveWeightedEmbeddings: Error: No valid SDClipModel instances found.")
            return {"ui": {"text": ["Error: No valid SDClipModel instances found."]}}

        tensors_to_save = {}
        for key_suffix, sd_clip_model_inst in sd_clip_instances.items():
            current_prompt_segments_with_weights = None
            if isinstance(tokenized_text_with_weights, dict):
                current_prompt_segments_with_weights = tokenized_text_with_weights.get(key_suffix)
            elif key_suffix == 'l' and not isinstance(tokenized_text_with_weights, dict):
                current_prompt_segments_with_weights = tokenized_text_with_weights
            
            if not current_prompt_segments_with_weights:
                print(f"SaveWeightedEmbeddings: Warning: No token data for '{key_suffix}'. Skipping.")
                continue

            if not hasattr(sd_clip_model_inst, 'transformer') or sd_clip_model_inst.transformer is None:
                 print(f"SaveWeightedEmbeddings: Error: SDClipModel for '{key_suffix}' has invalid 'transformer'. Skipping.")
                 continue
            device = sd_clip_model_inst.transformer.get_input_embeddings().weight.device

            prompt_token_id_segments = []
            max_segment_len = 0
            for segment_w_weights in current_prompt_segments_with_weights:
                prompt_token_id_segments.append([item[0] for item in segment_w_weights])
                max_segment_len = max(max_segment_len, len(segment_w_weights))
            
            if not prompt_token_id_segments:
                print(f"SaveWeightedEmbeddings: Warning: No token ID segments for '{key_suffix}'. Skipping.")
                continue

            empty_token_sequence = []
            if hasattr(sd_clip_model_inst, "gen_empty_tokens") and hasattr(sd_clip_model_inst, "special_tokens"):
                empty_token_sequence = sd_clip_model_inst.gen_empty_tokens(sd_clip_model_inst.special_tokens, max_segment_len)
            else:
                pad_token_id = None
                if hasattr(sd_clip_model_inst, "special_tokens") and "pad" in sd_clip_model_inst.special_tokens:
                    pad_token_id = sd_clip_model_inst.special_tokens["pad"]
                
                if pad_token_id is not None:
                    print(f"SaveWeightedEmbeddings: Warning: '{key_suffix}' SDClipModel instance missing 'gen_empty_tokens' or 'special_tokens'. Using simple pad token sequence for empty.")
                    empty_token_sequence = [pad_token_id] * max_segment_len
                else:
                    print(f"SaveWeightedEmbeddings: Error: Cannot determine pad token for '{key_suffix}'. Skipping weighting for this part.")
                    continue

            all_token_sequences_for_process_tokens = prompt_token_id_segments + [empty_token_sequence]
            embeds_combined, attention_mask_combined, _ = sd_clip_model_inst.process_tokens(
                all_token_sequences_for_process_tokens, device
            )

            num_prompt_segments = len(prompt_token_id_segments)
            prompt_embeds_full_segments = embeds_combined[:num_prompt_segments]
            embeds_empty_segment_full = embeds_combined[num_prompt_segments:]
            prompt_attention_mask_full_segments = attention_mask_combined[:num_prompt_segments]

            all_weighted_actual_embeddings_for_this_clip_part = []
            for s_idx in range(num_prompt_segments):
                current_segment_raw_embeds = prompt_embeds_full_segments[s_idx]
                segment_data_from_tokenizer = current_prompt_segments_with_weights[s_idx]
                current_segment_mask_from_process_tokens = prompt_attention_mask_full_segments[s_idx]
                weighted_segment_embeds = torch.zeros_like(current_segment_raw_embeds)

                for token_idx in range(current_segment_raw_embeds.shape[0]):
                    is_active_token = current_segment_mask_from_process_tokens[token_idx] == 1
                    original_embedding = current_segment_raw_embeds[token_idx]
                    embed_empty_for_this_pos = embeds_empty_segment_full[0, token_idx]

                    if is_active_token:
                        weight = segment_data_from_tokenizer[token_idx][1] if token_idx < len(segment_data_from_tokenizer) else 1.0
                        if weight == 1.0:
                            weighted_segment_embeds[token_idx] = original_embedding
                        else:
                            weighted_segment_embeds[token_idx] = (original_embedding - embed_empty_for_this_pos) * weight + embed_empty_for_this_pos
                    else:
                        weighted_segment_embeds[token_idx] = original_embedding
                
                actual_weighted_embeddings_this_segment = weighted_segment_embeds[current_segment_mask_from_process_tokens == 1]
                if actual_weighted_embeddings_this_segment.shape[0] > 0:
                    all_weighted_actual_embeddings_for_this_clip_part.append(actual_weighted_embeddings_this_segment)
            
            if not all_weighted_actual_embeddings_for_this_clip_part:
                print(f"SaveWeightedEmbeddings: Warning: No actual weighted tokens for '{key_suffix}'. Skipping.")
                continue

            final_embeddings_for_this_clip_part = torch.cat(all_weighted_actual_embeddings_for_this_clip_part, dim=0)
            
            # --- CORRECTED KEY NAMING ---
            if key_suffix == 't5xxl':
                save_key_in_file = key_suffix  # Use 't5xxl' directly
            else:
                save_key_in_file = f"clip_{key_suffix}" # Use 'clip_l', 'clip_g'
            # --- END CORRECTION ---

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveWeightedEmbeddings: Processed weighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveWeightedEmbeddings: Error: No weighted embeddings generated.")
            return {"ui": {"text": ["Error: No weighted embeddings generated."]}}

        subfolder_in_prefix, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0] 
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
        if subfolder_in_prefix:
            os.makedirs(full_output_folder, exist_ok=True)

        metadata = {}
        counter = 1
        max_counter = 99999
        while True:
            save_filename = f"{filename_base}_{counter:05}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)
            if not os.path.exists(save_path):
                break
            counter += 1
            if counter > max_counter:
                print(f"SaveWeightedEmbeddings: Warning: Max file attempts for {filename_base}.")
                return {"ui": {"text": [f"Error: Max files for {filename_base}"]}}
        
        print(f"SaveWeightedEmbeddings: Saving weighted to: {save_path}")
        comfy.utils.save_torch_file(tensors_to_save, save_path, metadata=metadata)
        return {"ui": {"text": [f"Saved weighted to {save_path}"]}}

class SaveA1111WeightedEmbeddings:
    def __init__(self):
        self.output_dir_list = folder_paths.get_folder_paths("embeddings")
        if not self.output_dir_list:
            print("SaveA1111WeightedEmbeddings: Warning: 'embeddings' folder type not found. Falling back to main output directory.")
            self.output_dir_list = [folder_paths.get_output_directory()]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "filename_prefix": ("STRING", {"default": "a1111_weighted_embed"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_a1111_weighted_embeddings"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit"

    def save_a1111_weighted_embeddings(self, clip, text, filename_prefix):
        if clip is None:
            raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveA1111WeightedEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        tokenized_text_with_weights = clip.tokenize(text)
        actual_clip_model_wrapper = clip.cond_stage_model
        
        sd_clip_instances = {}
        potential_clip_parts = {'l': 'clip_l', 'g': 'clip_g', 't5xxl': 't5xxl'}

        for key_suffix, attr_name in potential_clip_parts.items():
            if hasattr(actual_clip_model_wrapper, attr_name):
                model_instance = getattr(actual_clip_model_wrapper, attr_name)
                if model_instance is not None and hasattr(model_instance, "process_tokens") and hasattr(model_instance, "transformer"):
                    sd_clip_instances[key_suffix] = model_instance
                else:
                    if model_instance is None:
                        print(f"SaveA1111WeightedEmbeddings: Warning: Attribute '{attr_name}' is None. Skipping '{key_suffix}'.")
                    else: print(f"SaveA1111WeightedEmbeddings: Warning: Attribute '{attr_name}' (type {type(model_instance)}) is not a valid SDClipModel. Skipping '{key_suffix}'.")
        
        if not sd_clip_instances and hasattr(actual_clip_model_wrapper, "process_tokens") and hasattr(actual_clip_model_wrapper, "transformer"):
            print(f"SaveA1111WeightedEmbeddings: Warning: No standard attributes. Fallback: cond_stage_model as 'l'.")
            sd_clip_instances['l'] = actual_clip_model_wrapper
        
        if not sd_clip_instances:
            print(f"SaveA1111WeightedEmbeddings: Error: No valid SDClipModel instances found.")
            return {"ui": {"text": ["Error: No valid SDClipModel instances found."]}}

        tensors_to_save = {}
        for key_suffix, sd_clip_model_inst in sd_clip_instances.items():
            current_prompt_segments_with_weights = None
            if isinstance(tokenized_text_with_weights, dict):
                current_prompt_segments_with_weights = tokenized_text_with_weights.get(key_suffix)
            elif key_suffix == 'l' and not isinstance(tokenized_text_with_weights, dict):
                current_prompt_segments_with_weights = tokenized_text_with_weights
            
            if not current_prompt_segments_with_weights:
                print(f"SaveA1111WeightedEmbeddings: Warning: No token data for '{key_suffix}'. Skipping.")
                continue

            if not hasattr(sd_clip_model_inst, 'transformer') or sd_clip_model_inst.transformer is None:
                 print(f"SaveA1111WeightedEmbeddings: Error: SDClipModel for '{key_suffix}' has invalid 'transformer'. Skipping.")
                 continue
            device = sd_clip_model_inst.transformer.get_input_embeddings().weight.device

            prompt_token_id_segments = []
            for segment_w_weights in current_prompt_segments_with_weights:
                prompt_token_id_segments.append([item[0] for item in segment_w_weights])
            
            if not prompt_token_id_segments:
                print(f"SaveA1111WeightedEmbeddings: Warning: No token ID segments for '{key_suffix}'. Skipping.")
                continue

            embeds_out_full, attention_mask_full, _ = sd_clip_model_inst.process_tokens(
                prompt_token_id_segments, device
            )

            all_scaled_actual_embeddings_for_this_clip_part = []
            num_prompt_segments = embeds_out_full.shape[0]

            for s_idx in range(num_prompt_segments):
                current_segment_raw_embeds = embeds_out_full[s_idx] # Shape: [max_segment_len, dim]
                segment_data_from_tokenizer = current_prompt_segments_with_weights[s_idx]
                current_segment_mask_from_process_tokens = attention_mask_full[s_idx] # Shape: [max_segment_len]

                scaled_segment_embeds = torch.zeros_like(current_segment_raw_embeds)

                for token_idx in range(current_segment_raw_embeds.shape[0]): # Iterate up to max_segment_len
                    is_active_token = current_segment_mask_from_process_tokens[token_idx] == 1
                    original_embedding = current_segment_raw_embeds[token_idx]

                    if is_active_token:
                        weight = segment_data_from_tokenizer[token_idx][1] if token_idx < len(segment_data_from_tokenizer) else 1.0
                        
                        scaled_segment_embeds[token_idx] = original_embedding * weight
                    else:
                        scaled_segment_embeds[token_idx] = original_embedding
                
                actual_scaled_embeddings_this_segment = scaled_segment_embeds[current_segment_mask_from_process_tokens == 1]
                
                if actual_scaled_embeddings_this_segment.shape[0] > 0:
                    all_scaled_actual_embeddings_for_this_clip_part.append(actual_scaled_embeddings_this_segment)
            
            if not all_scaled_actual_embeddings_for_this_clip_part:
                print(f"SaveA1111WeightedEmbeddings: Warning: No actual scaled tokens found for '{key_suffix}' after processing. Skipping.")
                continue

            final_embeddings_for_this_clip_part = torch.cat(all_scaled_actual_embeddings_for_this_clip_part, dim=0)
            
            if key_suffix == 't5xxl':
                save_key_in_file = key_suffix
            else:
                save_key_in_file = f"clip_{key_suffix}"

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveA1111WeightedEmbeddings: Processed A1111-style weighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveA1111WeightedEmbeddings: Error: No A1111-style weighted embeddings generated.")
            return {"ui": {"text": ["Error: No A1111-style weighted embeddings generated."]}}

        subfolder_in_prefix, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0] 
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
        if subfolder_in_prefix:
            os.makedirs(full_output_folder, exist_ok=True)

        metadata = {}
        counter = 1
        max_counter = 99999
        while True:
            save_filename = f"{filename_base}_{counter:05}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)
            if not os.path.exists(save_path):
                break
            counter += 1
            if counter > max_counter:
                print(f"SaveA1111WeightedEmbeddings: Warning: Max file attempts for {filename_base}.")
                return {"ui": {"text": [f"Error: Max files for {filename_base}"]}}
        
        print(f"SaveA1111WeightedEmbeddings: Saving A1111-style weighted to: {save_path}")
        comfy.utils.save_torch_file(tensors_to_save, save_path, metadata=metadata)
        return {"ui": {"text": [f"Saved A1111-style weighted to {save_path}"]}}

NODE_CLASS_MAPPINGS = {
    "SaveTokenEmbeddings": SaveTokenEmbeddings,
    "SaveWeightedEmbeddings": SaveWeightedEmbeddings,
    "SaveA1111WeightedEmbeddings": SaveA1111WeightedEmbeddings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTokenEmbeddings": "Save Token Embeddings",
    "SaveWeightedEmbeddings": "Save Weighted Embeddings",
    "SaveA1111WeightedEmbeddings": "Save A1111-style Weighted Embeddings"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']