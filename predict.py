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
        self.model.device("cuda:0")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="Explain it in one sentence."),
        image: Path = Input(description="Input image"),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=100),
        temperature: float = Input(description="Temperature for sampling", default=0.7),
        top_p: float = Input(description="Top p for sampling", default=0.95),
    ) -> str:
        """Run a single prediction on the model"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        image = Image.open(image)
        image_tensor = self.model.image_preprocess(image).device("cuda:0")
        
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