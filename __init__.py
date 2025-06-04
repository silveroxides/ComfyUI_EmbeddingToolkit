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
        os.makedirs(self.output_dir_list[0], exist_ok=True)


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
        potential_clip_parts = {'l': 'clip_l', 'g': 'clip_g', 'pile_t5xl': 'pile_t5xl', 't5xl': 't5xl', 't5xxl': 't5xxl', 'umt5xxl': 'umt5xxl', 't5base': 't5base'}

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
                actual_embeddings_this_segment_masked = current_segment_embeds[current_segment_mask == 1]

                processed_segment = actual_embeddings_this_segment_masked
                num_tokens_in_segment_after_mask = processed_segment.shape[0]

                if num_tokens_in_segment_after_mask > 0:
                    if key_suffix == 'l' or key_suffix == 'g': # CLIP models (BOS and EOS)
                        if num_tokens_in_segment_after_mask >= 2:
                            processed_segment = processed_segment[1:-1]
                        else:
                            processed_segment = processed_segment[0:0]
                    elif key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'): # T5 models (EOS only)
                        if num_tokens_in_segment_after_mask >= 1:
                            processed_segment = processed_segment[:-1]
                        else:
                            processed_segment = processed_segment[0:0]

                if processed_segment.shape[0] > 0:
                    all_actual_embeddings_for_this_clip_part.append(processed_segment)

            if not all_actual_embeddings_for_this_clip_part:
                print(f"SaveTokenEmbeddings: Warning: No actual tokens for '{key_suffix}' after special token removal. Skipping.")
                continue

            final_embeddings_for_this_clip_part = torch.cat(all_actual_embeddings_for_this_clip_part, dim=0)

            if key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'):
                save_key_in_file = key_suffix
            else:
                save_key_in_file = f"clip_{key_suffix}"

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveTokenEmbeddings: Processed unweighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveTokenEmbeddings: Error: No unweighted embeddings generated.")
            return {"ui": {"text": ["Error: No unweighted embeddings generated."]}}

        subfolder_in_prefix, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0]
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)

        os.makedirs(full_output_folder, exist_ok=True) # Ensures the target folder exists

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
        os.makedirs(self.output_dir_list[0], exist_ok=True)

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
        if clip is None: raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveWeightedEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        tokenized_text_with_weights = clip.tokenize(text)
        actual_clip_model_wrapper = clip.cond_stage_model

        sd_clip_instances = {}
        potential_clip_parts = {'l': 'clip_l', 'g': 'clip_g', 'pile_t5xl': 'pile_t5xl', 't5xl': 't5xl', 't5xxl': 't5xxl', 'umt5xxl': 'umt5xxl', 't5base': 't5base'}

        for key_suffix, attr_name in potential_clip_parts.items():
            if hasattr(actual_clip_model_wrapper, attr_name):
                model_instance = getattr(actual_clip_model_wrapper, attr_name)
                if model_instance is not None and hasattr(model_instance, "process_tokens") and hasattr(model_instance, "transformer"):
                    sd_clip_instances[key_suffix] = model_instance

        if not sd_clip_instances and hasattr(actual_clip_model_wrapper, "process_tokens") and hasattr(actual_clip_model_wrapper, "transformer"):
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

            device = sd_clip_model_inst.transformer.get_input_embeddings().weight.device
            prompt_token_id_segments = [[item[0] for item in seg] for seg in current_prompt_segments_with_weights]
            if not prompt_token_id_segments: continue

            max_len_for_empty = getattr(sd_clip_model_inst, 'max_length', 77)
            empty_token_sequence = []
            if hasattr(sd_clip_model_inst, "gen_empty_tokens"):
                empty_token_sequence = sd_clip_model_inst.gen_empty_tokens(max_len_for_empty)
            else:
                pad_token_id = getattr(sd_clip_model_inst, 'pad_token_id', None)
                if pad_token_id is None and hasattr(sd_clip_model_inst, "special_tokens") and "pad" in sd_clip_model_inst.special_tokens:
                    pad_token_id = sd_clip_model_inst.special_tokens["pad"]
                if pad_token_id is not None:
                    empty_token_sequence = [pad_token_id] * max_len_for_empty
                else:
                    print(f"SaveWeightedEmbeddings: Error: Cannot determine pad/empty token for '{key_suffix}'. Skipping weighting for this part.")
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
                content_token_offset = 1 if (key_suffix == 'l' or key_suffix == 'g') else 0

                for token_idx_in_padded_seq in range(current_segment_raw_embeds.shape[0]):
                    is_active_token_in_mask = current_segment_mask_from_process_tokens[token_idx_in_padded_seq] == 1
                    original_embedding_at_pos = current_segment_raw_embeds[token_idx_in_padded_seq]
                    embed_empty_for_this_pos = embeds_empty_segment_full[0, token_idx_in_padded_seq]

                    if is_active_token_in_mask:
                        weight = 1.0
                        original_token_idx_for_weight = token_idx_in_padded_seq - content_token_offset
                        if original_token_idx_for_weight >= 0 and original_token_idx_for_weight < len(segment_data_from_tokenizer):
                            weight = segment_data_from_tokenizer[original_token_idx_for_weight][1]

                        if weight == 1.0:
                            weighted_segment_embeds[token_idx_in_padded_seq] = original_embedding_at_pos
                        else:
                            weighted_segment_embeds[token_idx_in_padded_seq] = (original_embedding_at_pos - embed_empty_for_this_pos) * weight + embed_empty_for_this_pos
                    else:
                        weighted_segment_embeds[token_idx_in_padded_seq] = original_embedding_at_pos

                actual_weighted_embeddings_this_segment_masked = weighted_segment_embeds[current_segment_mask_from_process_tokens == 1]

                processed_segment = actual_weighted_embeddings_this_segment_masked
                num_tokens_in_segment_after_mask = processed_segment.shape[0]

                if num_tokens_in_segment_after_mask > 0:
                    if key_suffix == 'l' or key_suffix == 'g':
                        if num_tokens_in_segment_after_mask >= 2: processed_segment = processed_segment[1:-1]
                        else: processed_segment = processed_segment[0:0]
                    elif key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'):
                        if num_tokens_in_segment_after_mask >= 1: processed_segment = processed_segment[:-1]
                        else: processed_segment = processed_segment[0:0]

                if processed_segment.shape[0] > 0:
                    all_weighted_actual_embeddings_for_this_clip_part.append(processed_segment)

            if not all_weighted_actual_embeddings_for_this_clip_part:
                print(f"SaveWeightedEmbeddings: Warning: No actual weighted tokens for '{key_suffix}' after special token removal. Skipping.")
                continue

            final_embeddings_for_this_clip_part = torch.cat(all_weighted_actual_embeddings_for_this_clip_part, dim=0)
            save_key_in_file = key_suffix if (key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile')) else f"clip_{key_suffix}"
            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveWeightedEmbeddings: Processed weighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveWeightedEmbeddings: Error: No weighted embeddings generated.")
            return {"ui": {"text": ["Error: No weighted embeddings generated."]}}

        subfolder_in_prefix, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0]
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
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
            self.output_dir_list = [folder_paths.get_output_directory()]
        os.makedirs(self.output_dir_list[0], exist_ok=True)

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
        if clip is None: raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveA1111WeightedEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        tokenized_text_with_weights = clip.tokenize(text)
        actual_clip_model_wrapper = clip.cond_stage_model

        sd_clip_instances = {}
        potential_clip_parts = {'l': 'clip_l', 'g': 'clip_g', 'pile_t5xl': 'pile_t5xl', 't5xl': 't5xl', 't5xxl': 't5xxl', 'umt5xxl': 'umt5xxl', 't5base': 't5base'}
        for key_suffix, attr_name in potential_clip_parts.items():
            if hasattr(actual_clip_model_wrapper, attr_name):
                model_instance = getattr(actual_clip_model_wrapper, attr_name)
                if model_instance is not None and hasattr(model_instance, "process_tokens") and hasattr(model_instance, "transformer"):
                    sd_clip_instances[key_suffix] = model_instance
        if not sd_clip_instances and hasattr(actual_clip_model_wrapper, "process_tokens") and hasattr(actual_clip_model_wrapper, "transformer"):
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

            if not current_prompt_segments_with_weights: continue

            device = sd_clip_model_inst.transformer.get_input_embeddings().weight.device
            prompt_token_id_segments = [[item[0] for item in seg] for seg in current_prompt_segments_with_weights]
            if not prompt_token_id_segments: continue

            embeds_out_full, attention_mask_full, _ = sd_clip_model_inst.process_tokens(
                prompt_token_id_segments, device
            )

            all_scaled_actual_embeddings_for_this_clip_part = []
            num_prompt_segments = embeds_out_full.shape[0]

            for s_idx in range(num_prompt_segments):
                current_segment_raw_embeds = embeds_out_full[s_idx]
                segment_data_from_tokenizer = current_prompt_segments_with_weights[s_idx]
                current_segment_mask_from_process_tokens = attention_mask_full[s_idx]

                scaled_segment_embeds = torch.zeros_like(current_segment_raw_embeds)
                content_token_offset = 1 if (key_suffix == 'l' or key_suffix == 'g') else 0

                for token_idx_in_padded_seq in range(current_segment_raw_embeds.shape[0]):
                    is_active_token_in_mask = current_segment_mask_from_process_tokens[token_idx_in_padded_seq] == 1
                    original_embedding_at_pos = current_segment_raw_embeds[token_idx_in_padded_seq]

                    if is_active_token_in_mask:
                        weight = 1.0
                        original_token_idx_for_weight = token_idx_in_padded_seq - content_token_offset
                        if original_token_idx_for_weight >= 0 and original_token_idx_for_weight < len(segment_data_from_tokenizer):
                            weight = segment_data_from_tokenizer[original_token_idx_for_weight][1]
                        scaled_segment_embeds[token_idx_in_padded_seq] = original_embedding_at_pos * weight
                    else:
                        scaled_segment_embeds[token_idx_in_padded_seq] = original_embedding_at_pos

                actual_scaled_embeddings_this_segment_masked = scaled_segment_embeds[current_segment_mask_from_process_tokens == 1]

                processed_segment = actual_scaled_embeddings_this_segment_masked
                num_tokens_in_segment_after_mask = processed_segment.shape[0]
                if num_tokens_in_segment_after_mask > 0:
                    if key_suffix == 'l' or key_suffix == 'g':
                        if num_tokens_in_segment_after_mask >= 2: processed_segment = processed_segment[1:-1]
                        else: processed_segment = processed_segment[0:0]
                    elif key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'):
                        if num_tokens_in_segment_after_mask >= 1: processed_segment = processed_segment[:-1]
                        else: processed_segment = processed_segment[0:0]

                if processed_segment.shape[0] > 0:
                    all_scaled_actual_embeddings_for_this_clip_part.append(processed_segment)

            if not all_scaled_actual_embeddings_for_this_clip_part: continue

            final_embeddings_for_this_clip_part = torch.cat(all_scaled_actual_embeddings_for_this_clip_part, dim=0)
            save_key_in_file = key_suffix if (key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile')) else f"clip_{key_suffix}"
            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveA1111WeightedEmbeddings: Processed A1111-style for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveA1111WeightedEmbeddings: Error: No A1111-style weighted embeddings generated.")
            return {"ui": {"text": ["Error: No A1111-style weighted embeddings generated."]}}

        subfolder_in_prefix, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0]
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
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


class SliceExistingEmbedding:
    def __init__(self):
        # Get the primary "embeddings" directory
        self.embeddings_dir_paths = folder_paths.get_folder_paths("embeddings")
        if not self.embeddings_dir_paths:
            print("SliceExistingEmbedding: Warning: 'embeddings' folder type not found. Falling back to main output directory.")
            self.embeddings_dir_paths = [folder_paths.get_output_directory()]
        self.primary_embeddings_dir = self.embeddings_dir_paths[0]
        os.makedirs(self.primary_embeddings_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        # Use folder_paths.get_filename_list to populate choices for the dropdown
        try:
            # Get filenames relative to the "embeddings" type paths
            embedding_files = [f for f in folder_paths.get_filename_list("embeddings") if f.lower().endswith((".safetensors", ".pt"))]
        except Exception as e:
            print(f"SliceExistingEmbedding: Error listing embedding files: {e}")
            embedding_files = ["None"]

        if not embedding_files: # Ensure there's at least one option
            embedding_files = ["None"]

        return {
            "required": {
                "embedding_file": (embedding_files,), # Dropdown with detected embedding files
                "output_filename_prefix": ("STRING", {"default": "sliced_embedding"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "slice_and_save"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit/Utils"

    def slice_and_save(self, embedding_file, output_filename_prefix):
        if embedding_file == "None" or not embedding_file:
            print("SliceExistingEmbedding: No embedding file selected.")
            return {"ui": {"text": ["No embedding file selected."]}}

        # Resolve the full path to the selected embedding file
        # folder_paths.get_full_path expects a folder_type and a relative filename
        full_file_path = folder_paths.get_full_path("embeddings", embedding_file)

        if full_file_path is None or not os.path.exists(full_file_path):
            # Fallback if get_full_path returns None (e.g., file not in registered paths)
            # or if the file just doesn't exist (could happen if list is stale)
            # Try constructing path directly from primary_embeddings_dir as a last resort
            potential_path = os.path.join(self.primary_embeddings_dir, embedding_file)
            if os.path.exists(potential_path):
                full_file_path = potential_path
            else:
                print(f"SliceExistingEmbedding: Error: File not found: {embedding_file}")
                return {"ui": {"text": [f"Error: File not found: {embedding_file}"]}}

        try:
            loaded_data = comfy.utils.load_torch_file(full_file_path)
            # Safetensors can have metadata in a special key, or PT files might not.
            # We should preserve it if it exists.
            metadata_key = "__metadata__"
            file_metadata = {}
            if isinstance(loaded_data, dict) and metadata_key in loaded_data:
                file_metadata = loaded_data.pop(metadata_key)

        except Exception as e:
            print(f"SliceExistingEmbedding: Error loading {embedding_file}: {e}")
            return {"ui": {"text": [f"Error loading {embedding_file}: {e}"]}}

        if not isinstance(loaded_data, dict):
            print(f"SliceExistingEmbedding: Error: Loaded data from {embedding_file} is not a dictionary (unexpected format).")
            return {"ui": {"text": [f"Error: Unexpected format in {embedding_file}."]}}

        sliced_tensors = {}
        any_sliced = False

        for key, tensor in loaded_data.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
                sliced_tensors[key] = tensor
                continue

            original_shape = tensor.shape
            num_tokens = original_shape[0] if tensor.ndim > 0 else 0
            processed_tensor = tensor

            if num_tokens == 0:
                print(f"SliceExistingEmbedding: Tensor for key '{key}' is already empty (shape {original_shape}). Keeping as is.")
            elif key.startswith("clip_l") or key.startswith("clip_g"):
                print(f"SliceExistingEmbedding: Applying CLIP-style slicing (BOS & EOS) to key '{key}' (shape {original_shape})")
                if num_tokens >= 2:
                    processed_tensor = tensor[1:-1]
                    any_sliced = True
                elif num_tokens > 0 : # e.g. only 1 token
                    processed_tensor = tensor[0:0]
                    any_sliced = True
                print(f"  Resulting shape: {processed_tensor.shape}")
            elif key.startswith("t5") or key.startswith("umt5") or key.startswith("pile"):
                print(f"SliceExistingEmbedding: Applying T5-style slicing (EOS) to key '{key}' (shape {original_shape})")
                if num_tokens >= 1:
                    processed_tensor = tensor[:-1]
                    any_sliced = True
                # if num_tokens == 0, it remains tensor[0:0] effectively
                print(f"  Resulting shape: {processed_tensor.shape}")
            else:
                print(f"SliceExistingEmbedding: Info: Key '{key}' does not match known slicing patterns. Tensor will not be sliced.")

            sliced_tensors[key] = processed_tensor

        if not any_sliced:
            msg = f"No changes made to {embedding_file} (no applicable slicing rules or tensors already short/empty)."
            print(f"SliceExistingEmbedding: {msg}")
            return {"ui": {"text": [msg]}}

        # Standard filename saving logic (adopted from other save nodes)
        subfolder_in_prefix, filename_base_from_prefix = os.path.split(output_filename_prefix)

        # Output will be in the primary "embeddings" directory, respecting any subfolder in the prefix
        output_target_folder = os.path.join(self.primary_embeddings_dir, subfolder_in_prefix)
        os.makedirs(output_target_folder, exist_ok=True)

        _, input_ext = os.path.splitext(embedding_file) # Preserve original extension if desired, or default to .safetensors
        output_ext = input_ext if input_ext.lower() in [".safetensors", ".pt"] else ".safetensors"

        counter = 1
        max_counter = 99999
        final_save_path = ""
        while True:
            save_filename_numbered = f"{filename_base_from_prefix}_{counter:05}{output_ext}"
            current_save_path_candidate = os.path.join(output_target_folder, save_filename_numbered)

            if not os.path.exists(current_save_path_candidate):
                final_save_path = current_save_path_candidate
                break

            counter += 1
            if counter > max_counter:
                msg = f"Max file attempts for {output_filename_prefix}. Please check your embeddings folder or prefix."
                print(f"SliceExistingEmbedding: {msg}")
                return {"ui": {"text": [f"Error: {msg}"]}}

        try:
            # Add back metadata if it was present
            if file_metadata:
                sliced_tensors[metadata_key] = file_metadata
            comfy.utils.save_torch_file(sliced_tensors, final_save_path) # Let save_torch_file handle metadata if it's smart, else it's saved as a normal key
            msg = f"Saved sliced embedding to {final_save_path}"
            print(f"SliceExistingEmbedding: {msg}")
            return {"ui": {"text": [msg]}}
        except Exception as e:
            msg = f"Error saving to {final_save_path}: {e}"
            print(f"SliceExistingEmbedding: {msg}")
            return {"ui": {"text": [f"Error saving to {final_save_path}: {e}"]}}


NODE_CLASS_MAPPINGS = {
    "SaveTokenEmbeddings": SaveTokenEmbeddings,
    "SaveWeightedEmbeddings": SaveWeightedEmbeddings,
    "SaveA1111WeightedEmbeddings": SaveA1111WeightedEmbeddings,
    "SliceExistingEmbedding": SliceExistingEmbedding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTokenEmbeddings": "Save Token Embeddings",
    "SaveWeightedEmbeddings": "Save Weighted Embeddings",
    "SaveA1111WeightedEmbeddings": "Save A1111-style Weighted Embeddings",
    "SliceExistingEmbedding": "Slice Existing Embedding File",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']