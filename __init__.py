import torch
import os
import folder_paths
import comfy.utils
import logging
import re
from comfy.sd1_clip import token_weights, escape_important, unescape_important

def tokenize_preserving_weights(clip, text):
    """
    Custom tokenizer helper that bypasses 'disable_weights=True' enforced by some
    tokenizers (e.g., Flux/Mistral/Qwen) by manually implementing the tokenization loop
    and weight parsing, ensuring weights are always respected.
    """
    tokenizer = clip.tokenizer
    out = {}
    
    potential_parts = [
        'clip_l', 'clip_g', 'clip_h',
        't5xxl', 't5xl', 't5base', 'pile_t5xl', 'umt5xxl',
        'qwen25_7b', 'qwen25_3b', 'qwen3_4b', 'qwen3_8b', 'qwen3_2b', 'qwen3_06b',
        'mistral3_24b',
        'gemma2_2b', 'gemma3_4b', 'gemma3_12b',
        'jina_clip_2', 'gemma', 'jina', 'byt5_small',
        'llama'
    ]

    # Dynamic discovery for SD1Tokenizer subclasses (like Flux2, Klein)
    # which store the sub-tokenizer attribute name in 'self.clip'
    if hasattr(tokenizer, "clip") and isinstance(tokenizer.clip, str):
        if tokenizer.clip not in potential_parts:
            potential_parts.append(tokenizer.clip)
    
    found_any = False
    
    for attr in potential_parts:
        if hasattr(tokenizer, attr):
            sub_tok = getattr(tokenizer, attr)
            
            # Check for essential attributes to perform manual tokenization
            if hasattr(sub_tok, "tokenizer") and hasattr(sub_tok, "tokens_start"):
                try:
                    # Manual parsing to ensure weights are respected
                    # We skip the 'disable_weights' check entirely by implementing the logic ourselves
                    
                    # 1. Parse weights
                    text_processed = escape_important(text)
                    parsed_weights = token_weights(text_processed, 1.0)
                    
                    # 2. Tokenize segments
                    tokens = []
                    embedding_identifier = getattr(sub_tok, "embedding_identifier", "embedding:")
                    embedding_directory = getattr(sub_tok, "embedding_directory", None)
                    
                    for weighted_segment, weight in parsed_weights:
                        to_tokenize = unescape_important(weighted_segment)
                        
                        # Handle embeddings (copied from SDTokenizer logic)
                        split = re.split(' {0}|\n{0}'.format(embedding_identifier), to_tokenize)
                        to_tokenize_list = [split[0]]
                        for i in range(1, len(split)):
                            to_tokenize_list.append("{}{}".format(embedding_identifier, split[i]))
                        to_tokenize_list = [x for x in to_tokenize_list if x != ""]
                        
                        for word in to_tokenize_list:
                            # Embedding check
                            if word.startswith(embedding_identifier) and embedding_directory is not None:
                                if hasattr(sub_tok, "_try_get_embedding"):
                                    embedding_name = word[len(embedding_identifier):].strip('\n')
                                    embed, leftover = sub_tok._try_get_embedding(embedding_name)
                                    if embed is None:
                                        logging.warning(f"EmbeddingToolkit: warning, embedding:{embedding_name} does not exist, ignoring")
                                    else:
                                        if len(embed.shape) == 1:
                                            tokens.append([(embed, weight)])
                                        else:
                                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                                    
                                    if leftover != "":
                                        word = leftover
                                    else:
                                        continue
                            
                            # Tokenize
                            end = 999999999999
                            if getattr(sub_tok, "tokenizer_adds_end_token", False):
                                end = -1
                            
                            start = getattr(sub_tok, "tokens_start", 0)
                            
                            # HF Tokenizer call
                            hf_tokens = sub_tok.tokenizer(word)["input_ids"]
                            
                            # Slice
                            sliced = hf_tokens[start:end] if end != 999999999999 else hf_tokens[start:]
                            
                            tokens.append([(t, weight) for t in sliced])

                    # 3. Batching (replicating SDTokenizer logic)
                    batched_tokens = []
                    batch = []
                    
                    start_token = getattr(sub_tok, "start_token", None)
                    end_token = getattr(sub_tok, "end_token", None)
                    pad_token = getattr(sub_tok, "pad_token", 0)
                    max_length = getattr(sub_tok, "max_length", 77)
                    max_word_length = getattr(sub_tok, "max_word_length", 8)
                    pad_to_max_length = getattr(sub_tok, "pad_to_max_length", True)
                    min_length = getattr(sub_tok, "min_length", None)
                    min_padding = getattr(sub_tok, "min_padding", None)
                    pad_left = getattr(sub_tok, "pad_left", False) # Mistral uses left padding!
                    
                    if start_token is not None:
                        batch.append((start_token, 1.0, 0))
                        
                    for i, t_group in enumerate(tokens):
                        is_large = len(t_group) >= max_word_length
                        has_end_token = 1 if end_token is not None else 0
                        
                        while len(t_group) > 0:
                            if len(t_group) + len(batch) > max_length - has_end_token:
                                remaining_length = max_length - len(batch) - has_end_token
                                if is_large:
                                    batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                                    if end_token is not None:
                                        batch.append((end_token, 1.0, 0))
                                    t_group = t_group[remaining_length:]
                                else:
                                    if end_token is not None:
                                        batch.append((end_token, 1.0, 0))
                                    # Padding
                                    if pad_to_max_length:
                                        pad_amt = max_length - len(batch)
                                        if pad_left:
                                             for _ in range(pad_amt): batch.insert(0, (pad_token, 1.0, 0))
                                        else:
                                             batch.extend([(pad_token, 1.0, 0)] * pad_amt)
                                
                                batched_tokens.append(batch)
                                batch = []
                                if start_token is not None:
                                    batch.append((start_token, 1.0, 0))
                            else:
                                batch.extend([(t,w,i+1) for t,w in t_group])
                                t_group = []
                    
                    # Last batch
                    if end_token is not None:
                        batch.append((end_token, 1.0, 0))
                    
                    # Min padding
                    if min_padding is not None:
                        pad_amt = min_padding
                        if pad_left:
                             for _ in range(pad_amt): batch.insert(0, (pad_token, 1.0, 0))
                        else:
                             batch.extend([(pad_token, 1.0, 0)] * pad_amt)
                             
                    # Pad to max
                    if pad_to_max_length and len(batch) < max_length:
                        pad_amt = max_length - len(batch)
                        if pad_left:
                             for _ in range(pad_amt): batch.insert(0, (pad_token, 1.0, 0))
                        else:
                             batch.extend([(pad_token, 1.0, 0)] * pad_amt)
                             
                    # Min length
                    if min_length is not None and len(batch) < min_length:
                        pad_amt = min_length - len(batch)
                        if pad_left:
                             for _ in range(pad_amt): batch.insert(0, (pad_token, 1.0, 0))
                        else:
                             batch.extend([(pad_token, 1.0, 0)] * pad_amt)
                    
                    batched_tokens.append(batch)
                    
                    # Map key
                    key = attr
                    if attr.startswith("clip_") and len(attr) == 6:
                        key = attr[5:]
                        
                    out[key] = batched_tokens
                    found_any = True
                    
                except Exception as e:
                    print(f"EmbeddingToolkit: Error manually tokenizing for '{attr}': {e}")
                    import traceback
                    traceback.print_exc()

    if not found_any:
        return clip.tokenize(text, llama_template="{}")
        
    return out

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
                "slice_bos_eos": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "token_embeds"}),
                "split_by_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_token_embeddings"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit"

    def save_token_embeddings(self, clip, text, slice_bos_eos, filename_prefix, split_by_model=False):
        if clip is None:
            raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveTokenEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        tokenized_text_with_weights = clip.tokenize(text, llama_template="{}")
        actual_clip_model_wrapper = clip.cond_stage_model

        sd_clip_instances = {}
        potential_clip_parts = {
            'l': 'clip_l', 'g': 'clip_g', 'h': 'clip_h',
            'pile_t5xl': 'pile_t5xl', 't5xl': 't5xl', 't5xxl': 't5xxl', 'umt5xxl': 'umt5xxl', 't5base': 't5base',
            'qwen25_7b': 'qwen25_7b', 'qwen25_3b': 'qwen25_3b', 'qwen3_4b': 'qwen3_4b', 'qwen3_8b': 'qwen3_8b', 'qwen3_2b': 'qwen3_2b', 'qwen3_06b': 'qwen3_06b',
            'mistral3_24b': 'mistral3_24b',
            'gemma2_2b': 'gemma2_2b', 'gemma3_4b': 'gemma3_4b', 'gemma3_12b': 'gemma3_12b',
            'jina_clip_2': 'jina_clip_2', 'gemma': 'gemma', 'jina': 'jina', 'byt5': 'byt5_small',
            'llama': 'llama',
        }

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

            embeds_out_full, attention_mask_full, num_tokens, embeds_info = sd_clip_model_inst.process_tokens(
                tokens_for_process_tokens_method, device
            )
            print(f"Returned values: num_tokens={num_tokens}, embeds_info={embeds_info}")

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

            if slice_bos_eos:
                final_embeddings_for_this_clip_part = torch.cat(all_actual_embeddings_for_this_clip_part, dim=0)
    
                num_tokens_in_final = final_embeddings_for_this_clip_part.shape[0]
                if key_suffix == 'l' or key_suffix == 'g':
                    if num_tokens_in_final >= 2:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[1:-1]
                    elif num_tokens_in_final > 0:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[0:0]
                elif key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'):
                    if num_tokens_in_final >= 1:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[:-1]
            else:
                final_embeddings_for_this_clip_part = torch.cat(all_actual_embeddings_for_this_clip_part, dim=0)

            if final_embeddings_for_this_clip_part.shape[0] == 0:
                print(f"SaveTokenEmbeddings: Warning: No embeddings left for '{key_suffix}' after slicing. Skipping.")
                continue

            if key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile') or key_suffix.startswith('qwen') or key_suffix.startswith('mistral') or key_suffix.startswith('gemma') or key_suffix.startswith('jina') or key_suffix.startswith('byt5') or key_suffix.startswith('llama'):
                save_key_in_file = key_suffix
            else:
                save_key_in_file = f"clip_{key_suffix}"

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveTokenEmbeddings: Processed unweighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveTokenEmbeddings: Error: No unweighted embeddings generated.")
            return {"ui": {"text": ["Error: No unweighted embeddings generated."]}}

        subfolder_in_prefix, filename_base_orig = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0]
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
        if subfolder_in_prefix:
            os.makedirs(full_output_folder, exist_ok=True)

        dicts_to_save = []
        if split_by_model:
            for key, tensor in tensors_to_save.items():
                dicts_to_save.append({key: tensor})
        else:
            dicts_to_save.append(tensors_to_save)

        saved_paths = []
        metadata = {}

        for current_tensors in dicts_to_save:
            keys = sorted(list(current_tensors.keys()))
            current_filename_base = filename_base_orig
            if keys:
                current_filename_base = f"{'_'.join(keys)}_{filename_base_orig}"

            save_filename = f"{current_filename_base}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)

            if os.path.exists(save_path):
                counter = 1
                max_counter = 99999
                while True:
                    save_filename = f"{current_filename_base}_{counter:05}.safetensors"
                    save_path = os.path.join(full_output_folder, save_filename)
                    if not os.path.exists(save_path):
                        break
                    counter += 1
                    if counter > max_counter:
                        print(f"SaveTokenEmbeddings: Warning: Max file attempts for {current_filename_base}.")
                        return {"ui": {"text": [f"Error: Max files for {current_filename_base}"]}}

            print(f"SaveTokenEmbeddings: Saving unweighted to: {save_path}")
            comfy.utils.save_torch_file(current_tensors, save_path, metadata=metadata)
            saved_paths.append(save_path)

        return {"ui": {"text": [f"Saved unweighted to {', '.join(saved_paths)}"]}}


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
                "slice_bos_eos": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "weighted_embed"}),
                "split_by_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_weighted_embeddings"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit"

    def save_weighted_embeddings(self, clip, text, slice_bos_eos, filename_prefix, split_by_model=False):
        if clip is None:
            raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveWeightedEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        # Use custom tokenizer to ensure we get weights even for Flux/Mistral/Qwen
        tokenized_text_with_weights = tokenize_preserving_weights(clip, text)
        actual_clip_model_wrapper = clip.cond_stage_model

        sd_clip_instances = {}
        potential_clip_parts = {
            'l': 'clip_l', 'g': 'clip_g', 'h': 'clip_h',
            'pile_t5xl': 'pile_t5xl', 't5xl': 't5xl', 't5xxl': 't5xxl', 'umt5xxl': 'umt5xxl', 't5base': 't5base',
            'qwen25_7b': 'qwen25_7b', 'qwen25_3b': 'qwen25_3b', 'qwen3_4b': 'qwen3_4b', 'qwen3_8b': 'qwen3_8b', 'qwen3_2b': 'qwen3_2b', 'qwen3_06b': 'qwen3_06b',
            'mistral3_24b': 'mistral3_24b',
            'gemma2_2b': 'gemma2_2b', 'gemma3_4b': 'gemma3_4b', 'gemma3_12b': 'gemma3_12b',
            'jina_clip_2': 'jina_clip_2', 'gemma': 'gemma', 'jina': 'jina', 'byt5': 'byt5_small',
            'llama': 'llama',
        }

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
            embeds_combined, attention_mask_combined, num_tokens, embeds_info = sd_clip_model_inst.process_tokens(
                all_token_sequences_for_process_tokens, device
            )
            print(f"Returned values: num_tokens={num_tokens}, embeds_info={embeds_info}")

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

            if slice_bos_eos:
                final_embeddings_for_this_clip_part = torch.cat(all_weighted_actual_embeddings_for_this_clip_part, dim=0)
    
                num_tokens_in_final = final_embeddings_for_this_clip_part.shape[0]
                if key_suffix == 'l' or key_suffix == 'g':
                    if num_tokens_in_final >= 2:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[1:-1]
                    elif num_tokens_in_final > 0:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[0:0]
                elif key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'):
                    if num_tokens_in_final >= 1:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[:-1]
            else:
                final_embeddings_for_this_clip_part = torch.cat(all_weighted_actual_embeddings_for_this_clip_part, dim=0)

            if final_embeddings_for_this_clip_part.shape[0] == 0:
                print(f"SaveWeightedEmbeddings: Warning: No embeddings left for '{key_suffix}' after slicing. Skipping.")
                continue

            if key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile') or key_suffix.startswith('qwen') or key_suffix.startswith('mistral') or key_suffix.startswith('gemma') or key_suffix.startswith('jina') or key_suffix.startswith('byt5') or key_suffix.startswith('llama'):
                save_key_in_file = key_suffix
            else:
                save_key_in_file = f"clip_{key_suffix}"

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveWeightedEmbeddings: Processed weighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveWeightedEmbeddings: Error: No weighted embeddings generated.")
            return {"ui": {"text": ["Error: No weighted embeddings generated."]}}

        subfolder_in_prefix, filename_base_orig = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0]
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
        if subfolder_in_prefix:
            os.makedirs(full_output_folder, exist_ok=True)

        dicts_to_save = []
        if split_by_model:
            for key, tensor in tensors_to_save.items():
                dicts_to_save.append({key: tensor})
        else:
            dicts_to_save.append(tensors_to_save)

        saved_paths = []
        metadata = {}

        for current_tensors in dicts_to_save:
            keys = sorted(list(current_tensors.keys()))
            current_filename_base = filename_base_orig
            if keys:
                current_filename_base = f"{'_'.join(keys)}_{filename_base_orig}"

            save_filename = f"{current_filename_base}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)

            if os.path.exists(save_path):
                counter = 1
                max_counter = 99999
                while True:
                    save_filename = f"{current_filename_base}_{counter:05}.safetensors"
                    save_path = os.path.join(full_output_folder, save_filename)
                    if not os.path.exists(save_path):
                        break
                    counter += 1
                    if counter > max_counter:
                        print(f"SaveWeightedEmbeddings: Warning: Max file attempts for {current_filename_base}.")
                        return {"ui": {"text": [f"Error: Max files for {current_filename_base}"]}}

            print(f"SaveWeightedEmbeddings: Saving weighted to: {save_path}")
            comfy.utils.save_torch_file(current_tensors, save_path, metadata=metadata)
            saved_paths.append(save_path)

        return {"ui": {"text": [f"Saved weighted to {', '.join(saved_paths)}"]}}

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
                "slice_bos_eos": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "a1111_weighted_embed"}),
                "split_by_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_a1111_weighted_embeddings"
    OUTPUT_NODE = True
    CATEGORY = "EmbeddingToolkit"

    def save_a1111_weighted_embeddings(self, clip, text, slice_bos_eos, filename_prefix, split_by_model=False):
        if clip is None:
            raise RuntimeError("CLIP input is None.")
        if clip.cond_stage_model is None:
            print("SaveA1111WeightedEmbeddings: Error: clip.cond_stage_model is None.")
            return {"ui": {"text": ["Error: CLIP's cond_stage_model is None."]}}

        # Use custom tokenizer to ensure we get weights even for Flux/Mistral/Qwen
        tokenized_text_with_weights = tokenize_preserving_weights(clip, text)
        actual_clip_model_wrapper = clip.cond_stage_model

        sd_clip_instances = {}
        potential_clip_parts = {
            'l': 'clip_l', 'g': 'clip_g', 'h': 'clip_h',
            'pile_t5xl': 'pile_t5xl', 't5xl': 't5xl', 't5xxl': 't5xxl', 'umt5xxl': 'umt5xxl', 't5base': 't5base',
            'qwen25_7b': 'qwen25_7b', 'qwen25_3b': 'qwen25_3b', 'qwen3_4b': 'qwen3_4b', 'qwen3_8b': 'qwen3_8b', 'qwen3_2b': 'qwen3_2b', 'qwen3_06b': 'qwen3_06b',
            'mistral3_24b': 'mistral3_24b',
            'gemma2_2b': 'gemma2_2b', 'gemma3_4b': 'gemma3_4b', 'gemma3_12b': 'gemma3_12b',
            'jina_clip_2': 'jina_clip_2', 'gemma': 'gemma', 'jina': 'jina', 'byt5': 'byt5_small',
            'llama': 'llama',
        }

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

            embeds_out_full, attention_mask_full, num_tokens, embeds_info = sd_clip_model_inst.process_tokens(
                prompt_token_id_segments, device
            )
            print(f"Returned values: num_tokens={num_tokens}, embeds_info={embeds_info}")

            all_scaled_actual_embeddings_for_this_clip_part = []
            num_prompt_segments = embeds_out_full.shape[0]

            for s_idx in range(num_prompt_segments):
                current_segment_raw_embeds = embeds_out_full[s_idx]
                segment_data_from_tokenizer = current_prompt_segments_with_weights[s_idx]
                current_segment_mask_from_process_tokens = attention_mask_full[s_idx]

                scaled_segment_embeds = torch.zeros_like(current_segment_raw_embeds)

                for token_idx in range(current_segment_raw_embeds.shape[0]):
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

            if slice_bos_eos:
                final_embeddings_for_this_clip_part = torch.cat(all_scaled_actual_embeddings_for_this_clip_part, dim=0)
    
                num_tokens_in_final = final_embeddings_for_this_clip_part.shape[0]
                if key_suffix == 'l' or key_suffix == 'g':
                    if num_tokens_in_final >= 2:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[1:-1]
                    elif num_tokens_in_final > 0:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[0:0]
                elif key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile'):
                    if num_tokens_in_final >= 1:
                        final_embeddings_for_this_clip_part = final_embeddings_for_this_clip_part[:-1]
            else:
                final_embeddings_for_this_clip_part = torch.cat(all_scaled_actual_embeddings_for_this_clip_part, dim=0)

            if final_embeddings_for_this_clip_part.shape[0] == 0:
                print(f"SaveA1111WeightedEmbeddings: Warning: No embeddings left for '{key_suffix}' after slicing. Skipping.")
                continue

            if key_suffix.startswith('t5') or key_suffix.startswith('umt5') or key_suffix.startswith('pile') or key_suffix.startswith('qwen') or key_suffix.startswith('mistral') or key_suffix.startswith('gemma') or key_suffix.startswith('jina') or key_suffix.startswith('byt5') or key_suffix.startswith('llama'):
                save_key_in_file = key_suffix
            else:
                save_key_in_file = f"clip_{key_suffix}"

            tensors_to_save[save_key_in_file] = final_embeddings_for_this_clip_part.cpu()
            print(f"SaveA1111WeightedEmbeddings: Processed A1111-style weighted for '{save_key_in_file}': shape {final_embeddings_for_this_clip_part.shape}")

        if not tensors_to_save:
            print("SaveA1111WeightedEmbeddings: Error: No A1111-style weighted embeddings generated.")
            return {"ui": {"text": ["Error: No A1111-style weighted embeddings generated."]}}

        subfolder_in_prefix, filename_base_orig = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir_list[0]
        full_output_folder = os.path.join(primary_output_dir, subfolder_in_prefix)
        if subfolder_in_prefix:
            os.makedirs(full_output_folder, exist_ok=True)

        dicts_to_save = []
        if split_by_model:
            for key, tensor in tensors_to_save.items():
                dicts_to_save.append({key: tensor})
        else:
            dicts_to_save.append(tensors_to_save)

        saved_paths = []
        metadata = {}

        for current_tensors in dicts_to_save:
            keys = sorted(list(current_tensors.keys()))
            current_filename_base = filename_base_orig
            if keys:
                current_filename_base = f"{'_'.join(keys)}_{filename_base_orig}"

            save_filename = f"{current_filename_base}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)

            if os.path.exists(save_path):
                counter = 1
                max_counter = 99999
                while True:
                    save_filename = f"{current_filename_base}_{counter:05}.safetensors"
                    save_path = os.path.join(full_output_folder, save_filename)
                    if not os.path.exists(save_path):
                        break
                    counter += 1
                    if counter > max_counter:
                        print(f"SaveA1111WeightedEmbeddings: Warning: Max file attempts for {current_filename_base}.")
                        return {"ui": {"text": [f"Error: Max files for {current_filename_base}"]}}

            print(f"SaveA1111WeightedEmbeddings: Saving A1111-style weighted to: {save_path}")
            comfy.utils.save_torch_file(current_tensors, save_path, metadata=metadata)
            saved_paths.append(save_path)

        return {"ui": {"text": [f"Saved A1111-style weighted to {', '.join(saved_paths)}"]}}


class SliceExistingEmbedding:
    def __init__(self):
        self.embeddings_dir_paths = folder_paths.get_folder_paths("embeddings")
        if not self.embeddings_dir_paths:
            print("SliceExistingEmbedding: Warning: 'embeddings' folder type not found. Falling back to main output directory.")
            self.embeddings_dir_paths = [folder_paths.get_output_directory()]
        self.primary_embeddings_dir = self.embeddings_dir_paths[0]
        os.makedirs(self.primary_embeddings_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        try:
            embedding_files = [f for f in folder_paths.get_filename_list("embeddings") if f.lower().endswith((".safetensors", ".pt"))]
        except Exception as e:
            print(f"SliceExistingEmbedding: Error listing embedding files: {e}")
            embedding_files = ["None"]

        if not embedding_files:
            embedding_files = ["None"]

        return {
            "required": {
                "embedding_file": (embedding_files,),
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

        full_file_path = folder_paths.get_full_path("embeddings", embedding_file)

        if full_file_path is None or not os.path.exists(full_file_path):
            potential_path = os.path.join(self.primary_embeddings_dir, embedding_file)
            if os.path.exists(potential_path):
                full_file_path = potential_path
            else:
                print(f"SliceExistingEmbedding: Error: File not found: {embedding_file}")
                return {"ui": {"text": [f"Error: File not found: {embedding_file}"]}}

        try:
            loaded_data = comfy.utils.load_torch_file(full_file_path)
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
                elif num_tokens > 0 :
                    processed_tensor = tensor[0:0]
                    any_sliced = True
                print(f"  Resulting shape: {processed_tensor.shape}")
            elif key.startswith("t5") or key.startswith("umt5") or key.startswith("pile"):
                print(f"SliceExistingEmbedding: Applying T5-style slicing (EOS) to key '{key}' (shape {original_shape})")
                if num_tokens >= 1:
                    processed_tensor = tensor[:-1]
                    any_sliced = True
                print(f"  Resulting shape: {processed_tensor.shape}")
            else:
                print(f"SliceExistingEmbedding: Info: Key '{key}' does not match known slicing patterns. Tensor will not be sliced.")

            sliced_tensors[key] = processed_tensor

        if not any_sliced:
            msg = f"No changes made to {embedding_file} (no applicable slicing rules or tensors already short/empty)."
            print(f"SliceExistingEmbedding: {msg}")
            return {"ui": {"text": [msg]}}

        subfolder_in_prefix, filename_base_from_prefix = os.path.split(output_filename_prefix)

        keys = sorted([k for k in sliced_tensors.keys() if k != "__metadata__"])
        if keys:
            filename_base_from_prefix = f"{'_'.join(keys)}_{filename_base_from_prefix}"

        output_target_folder = os.path.join(self.primary_embeddings_dir, subfolder_in_prefix)
        os.makedirs(output_target_folder, exist_ok=True)

        _, input_ext = os.path.splitext(embedding_file)
        output_ext = input_ext if input_ext.lower() in [".safetensors", ".pt"] else ".safetensors"

        final_save_path = ""

        save_filename_numbered = f"{filename_base_from_prefix}{output_ext}"
        current_save_path_candidate = os.path.join(output_target_folder, save_filename_numbered)

        if not os.path.exists(current_save_path_candidate):
            final_save_path = current_save_path_candidate
        else:
            counter = 1
            max_counter = 99999
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
            if file_metadata:
                sliced_tensors[metadata_key] = file_metadata
            comfy.utils.save_torch_file(sliced_tensors, final_save_path)
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