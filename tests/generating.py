import torch
import argparse
from adapters import Tokenizer, TransformerLM, run_load_checkpoint, AdamW

def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device):
    model.eval()

    token_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            last_token_logits = logits[0, -1, :]
            last_token_logits /= temperature
            probs = torch.softmax(last_token_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            probs_to_sample_from = probs.clone()
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs_to_sample_from[indices_to_remove] = 0

            renormalized_probs = probs_to_sample_from / torch.sum(probs_to_sample_from)

            next_token_id = torch.multinomial(renormalized_probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token_id.unsqueeze(0)], dim=1)

            if next_token_id.item() == tokenizer.byte_to_id.get(b"<|endoftext|>"):
                break

    generated_ids = input_tensor[0].tolist()
    return tokenizer.decode(generated_ids)

def main():

    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer language model.")

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='The starting prompt.')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json'
    merges_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt'
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=256,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        device=device
    )
    model.to(device)
    
    # Instantiate a dummy optimizer for the loading function
    optimizer = AdamW(model.parameters())

    # Load the trained weights
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    run_load_checkpoint(args.checkpoint_path, model, optimizer)
    
    # --- 3. & 4. Generate and Print ---
    print("--- Prompt ---")
    print(args.prompt)
    print("\n--- Generation ---")
    generated_text = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, device)
    print(generated_text)


if __name__ == "__main__":
    main()