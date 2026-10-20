from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class LM:
    def __init__(self, model_name="google/gemma-3-4b-it"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        # 4-bit quantization allows the 4B model to fit in 6GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0" if self.device == "cuda" else "cpu",
        )
        print(f"Model loaded on: {self.model.device}")

    def generate_text(self, prompt: str, desired_output_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=desired_output_tokens,
            )

        # Slice to remove the prompt from the output
        return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
