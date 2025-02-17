import torch
import clip
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class LoRa_with_text():
    def __init__(self):
        self.pipe = self.init_pipe()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        print("Model LoRa loaded!")

    def init_pipe(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

        config = LoraConfig(r=16, lora_alpha=32, target_modules=["to_q", "to_v", "query", "value"], lora_dropout=0.1)
        pipe.unet = get_peft_model(pipe.unet, config)
        return pipe

    def generate_image(self, prompt, style_prompts, style_weights):
        with torch.no_grad():
            prompt_embeds = self.get_embedding(prompt)
            final_embeds = (1 - sum(style_weights)) * prompt_embeds
            for style_prompt, style_weight in zip(style_prompts, style_weights):
                style_embedding = self.get_embedding(style_prompt)
                final_embeds = final_embeds + style_weight * style_embedding
            image = self.pipe(prompt_embeds=final_embeds, num_inference_steps=50).images[0]
        return image

    def get_embedding(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids.to(
            device)
        with torch.no_grad():
            return self.text_encoder(tokens).last_hidden_state