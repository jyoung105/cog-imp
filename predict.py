# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "MILVLG/imp-v1-3b"
MODEL_CACHE = "model-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            trust_remote_code=True,
        )
        self.model.to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt"),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=100),
        temperature: float = Input(description="Temperature for sampling", default=0.7),
        top_p: float = Input(description="Top p for sampling", default=0.95),
    ) -> str:
        """Run a single prediction on the model"""
        prompt_template = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt}. ASSISTANT:"
        input_ids = self.tokenizer(prompt_template, return_tensors="pt").input_ids.to("cuda")
        image = Image.open(image)
        image_tensor = self.model.image_preprocess(image).to("cuda")
        
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            images=image_tensor,
            do_sample=True,
            use_cache=True,)[0]
        
        output = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        return output