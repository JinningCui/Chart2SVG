# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

# Ensure Chart2SVG can be imported
# Assuming the script is run from the project root or Chart2SVG directory
# We add the parent directory of 'models' (which is Chart2SVG) to sys.path just in case
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from Chart2SVG.data import (
        SVGToken, AttribMapper, ContainerMapper, GradientsMapper, PathCMDMapper,
        PathMapper, ShapeMapper, NUM_TOKEN, TokenDescMapper
    )
except ImportError:
    # Fallback if running from a different context where Chart2SVG is not resolved directly
    # This assumes the standard structure Chart2SVG/Chart2SVG exists
    sys.path.append(str(project_root.parent))
    from Chart2SVG.data import (
        SVGToken, AttribMapper, ContainerMapper, GradientsMapper, PathCMDMapper,
        PathMapper, ShapeMapper, NUM_TOKEN, TokenDescMapper
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Special tokens: Combine all mappers to get the list of SVG tokens
combined_mapper = {**PathMapper, **PathCMDMapper, **ShapeMapper, **ContainerMapper,
                   **GradientsMapper, **AttribMapper, **SVGToken}
SVG_TOKEN_LIST = sorted(list(set(combined_mapper.values())))

# Add NUM_TOKEN if not present
if NUM_TOKEN not in SVG_TOKEN_LIST:
    SVG_TOKEN_LIST.append(NUM_TOKEN)


def init_token_embedding(model, tokenizer, new_tokens, logger_instance):
    """
    Initializes embeddings for newly added tokens using semantic descriptions.
    Adapted from unsloth_llama3.py for Qwen-VL.
    """
    logger_instance.info("Attempting semantic initialization for new token embeddings...")

    embedding_layer = model.get_input_embeddings()
    if embedding_layer is None:
        logger_instance.error("Could not get input embedding layer from the model.")
        return model

    # We need to access the original weights. 
    # Since we just resized, the new rows are randomly initialized (usually).
    # We want to init them based on the mean of the description tokens' embeddings.
    # The description tokens should be within the *original* vocab range.
    
    # Current weight shape
    current_weights = embedding_layer.weight
    current_vocab_size = current_weights.shape[0]
    
    # We assume the new tokens were just added at the end.
    # But strictly speaking, we should look up their IDs.
    
    initialized_count = 0
    skipped_count = 0

    with torch.no_grad():
        for token_str in new_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            
            # Sanity check: ensure token_id is valid
            if token_id is None:
                logger_instance.warning(f"Token '{token_str}' has no ID. Skipping.")
                continue

            # Get description
            if token_str == NUM_TOKEN:
                desc = 'number coordinate value'
            else:
                # Try direct lookup first (keys in TokenDescMapper have brackets)
                desc = TokenDescMapper.get(token_str)
                if not desc:
                     # Fallback: try stripping brackets if not found
                     desc = TokenDescMapper.get(token_str.strip('[]'))

            if not desc:
                logger_instance.warning(f"No description found for token: '{token_str}'. Skipping semantic init.")
                skipped_count += 1
                continue

            # Tokenize description
            # For Qwen-VL, use the processor's tokenizer (passed as tokenizer here)
            # We want raw IDs without special tokens if possible, or just the content
            tokenized_output = tokenizer(desc, add_special_tokens=False)
            tokenized_ids = tokenized_output['input_ids']

            if not tokenized_ids:
                logger_instance.warning(f"Description for '{token_str}' is empty after tokenization. Skipping.")
                skipped_count += 1
                continue

            # Calculate mean embedding
            # We use the current embeddings to look up these IDs.
            # Assuming description words are common words, they are in the original vocab.
            try:
                desc_tensor = torch.tensor(tokenized_ids, device=embedding_layer.weight.device)
                mean_embedding = embedding_layer(desc_tensor).mean(dim=0)
                
                # Assign to the new token
                embedding_layer.weight[token_id, :] = mean_embedding
                initialized_count += 1
            except Exception as e:
                logger_instance.error(f"Error calculating/assigning embedding for '{token_str}': {e}")
                skipped_count += 1

    logger_instance.info(
        f"Semantic initialization finished. Initialized: {initialized_count}, Skipped: {skipped_count}")
    return model


def main():
    # Configuration
    MODEL_PATH = "models/Qwen3-VL-4B-Instruct"
    SAVE_PATH = "models/Qwen3_Chart_4B_Initialized"
    
    logger.info(f"Loading processor from {MODEL_PATH}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
        tokenizer = processor.tokenizer
    except Exception as e:
        logger.error(f"Failed to load AutoProcessor: {e}. Trying AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
        processor = None # If there is no processor, we might be in a text-only mode or manual handling

    logger.info(f"Loading model from {MODEL_PATH}...")
    # Load model to CPU first to save memory during init, or directly to GPU if needed.
    # Using 'auto' device map.
    device_map = {"": "cuda:1"}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        device_map=device_map,
        torch_dtype="auto"
    )

    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocab size: {original_vocab_size}")

    # 1. Add New Tokens
    logger.info(f"Adding {len(SVG_TOKEN_LIST)} SVG tokens to tokenizer...")
    num_added = tokenizer.add_tokens(SVG_TOKEN_LIST)
    logger.info(f"Actually added {num_added} tokens.")

    if num_added > 0:
        # 2. Resize Model Embeddings
        new_vocab_size = len(tokenizer)
        logger.info(f"Resizing model embeddings from {original_vocab_size} to {new_vocab_size}...")
        model.resize_token_embeddings(new_vocab_size)

        # 3. Semantic Initialization
        # We only need to initialize the *newly added* tokens.
        # We can identify them by checking which tokens in SVG_TOKEN_LIST were actually added 
        # (or just try to init all SVG tokens, overwriting is fine if they are special tokens we control).
        # But efficiently, we iterate over SVG_TOKEN_LIST.
        init_token_embedding(model, tokenizer, SVG_TOKEN_LIST, logger)
    else:
        logger.info("No tokens added. Skipping resize and initialization.")

    # 4. Save
    logger.info(f"Saving initialized model and processor to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    if processor:
        processor.save_pretrained(SAVE_PATH)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
