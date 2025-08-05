import os
import torch
import argparse
import einops
import numpy as np
from adapters import Tokenizer, AdamW, TransformerLM, run_cross_entropy, run_get_batch, run_save_checkpoint, run_load_checkpoint

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json'
    merges_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt'
    my_tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])
    vocab_size = len(my_tokenizer.vocab)

    input_text_path = '/home/minko/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt'
    output_numpy_path = '/home/minko/cs336/assignment1-basics/tests/fixtures/tinyStories_train.npy'

    with open(input_text_path, 'r', encoding='utf-8') as f:
        token_ids_iterator = my_tokenizer.encode_iterable(f)
        # Create a single large NumPy array from the iterator of token IDs
        tokenized_dataset = np.fromiter(token_ids_iterator, dtype=np.uint16)

    np.save(output_numpy_path, tokenized_dataset)
    train_data = np.load(output_numpy_path, mmap_mode='r')

    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training.')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads.')
    parser.add_argument('--d_ff', type=int, default=1344, help='Dimension of the feed-forward layer.')
    parser.add_argument('--rope_theta', type=int, default=10000, help='RoPE theta.')
    parser.add_argument('--context_length', type=int, default=256, help='The context length of the model.')
    parser.add_argument('--num_training_steps', type=int, default=100000, help='Number of training steps.')
    parser.add_argument('--log_interval', type=int, default=1000, help='How often to log training loss.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='How often to save a checkpoint.')

    args = parser.parse_args()  

    print(f"Starting training with the following hyperparameters:")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Model Dimension: {args.d_model}")
    print(f"  - Layers: {args.num_layers}")
    print(f"  - Heads: {args.num_heads}")
    print(f"  - Context length: {args.context_length}")
    print(f"  - Log interval: {args.log_interval}")
    print(f"  - Checkpoint interval: {args.checkpoint_interval}")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for step in range(args.num_training_steps):
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device='cuda')
        logits = model(x)
        
        logits_rearranged = einops.rearrange(logits, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size")
        targets_rearranged = einops.rearrange(y, "batch_size seq_len -> (batch_size seq_len)")
        loss = run_cross_entropy(logits_rearranged, targets_rearranged)

        # 1. Clear the gradients from the previous step
        optimizer.zero_grad()
        # 2. Compute the gradients for this step
        loss.backward()
        # 3. Update the model's weights
        optimizer.step()
        
        if step % args.log_interval == 0:
            print(f"Step {step}/{args.num_training_steps} | Loss: {loss.item():.4f}")

        if step > 0 and step % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_step_{step}.pt"
            print(f"Saving checkpoint to {checkpoint_path}...")
            run_save_checkpoint(model, optimizer, step, checkpoint_path)

if __name__ == "__main__":
    main()
