import torch
import clip
from typing import List, Optional
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from Interpolation_styles import InterpolationStyles
from StyleEmbedder import StyleEmbedder

class StableDiffusionWithText:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/mnt/c/Models" if torch.cuda.is_available() else "D:/Models"

    def __init__(self, use_lora: bool = True, model_id: float = 1.5,
                 projection_model_path: str = "style_projection_model.pt"):
        self.name: str = "StableDiffusion"
        self.use_lora: bool = use_lora
        self.model_id: float = model_id
        self.pipe = self._initialize_pipeline()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.DEVICE)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=self.MODEL_PATH
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=self.MODEL_PATH
        ).to(self.DEVICE)
        self.embedder = StyleEmbedder(
            projection_model_path=projection_model_path,
            clip_model=self.clip_model,
            clip_preprocess=self.preprocess,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=self.DEVICE
        )
        print(f"Model {self.name} loaded on {self.DEVICE}")

    def _initialize_pipeline(self) -> StableDiffusionPipeline:
        if self.model_id == 1.5:
            model_name = "runwayml/stable-diffusion-v1-5"
            self.name += " 1.5"
        elif self.model_id == 1.4:
            model_name = "CompVis/stable-diffusion-v1-4"
            self.name += " 1.4"
        else:
            raise ValueError("Unsupported model_id: {self.model_id}. Expected 1.5 or 1.4")
        dtype = torch.float16 if self.DEVICE == "cuda" else torch.float32
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                cache_dir=self.MODEL_PATH,
                torch_dtype=dtype
            ).to(self.DEVICE)

            if self.use_lora:
                model_lora_path = "sayakpaul/sd-model-finetuned-lora-t4"
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights(model_lora_path)
                self.name += " with LoRA"
            return pipe
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize pipeline {model_name}: {str(e)}")

    def generate_image(self,
                       prompt:str,
                       style_prompts: Optional[List[str]] = None,
                       style_folders: Optional[List[str]] = None,
                       style_weights: Optional[List[float]] = None,
                       interpolation_type: str = "linear",
                       swirl_factor=None,
                       smoothness=None,
                       seed=None) -> torch.Tensor:
        if style_prompts is None:
            style_prompts = []
        if style_weights is None:
            style_weights = []
        if style_folders is None:
            style_folders = []

        if len(style_prompts) + len(style_folders) != len(style_weights):
            raise ValueError("Style prompts must have same number of weights")

        with torch.no_grad():
            prompt_embeddings = self._get_embeddings(prompt)
            if style_prompts or style_folders:
                styles_embeddings = []
                styles_embeddings.extend(self._get_embeddings(style_prompt) for style_prompt in style_prompts)
                styles_embeddings.extend(self.embedder.get_embedding(style_folder=folder) for folder in style_folders)
                interpolation = InterpolationStyles(prompt_embeddings, styles_embeddings, style_weights)
                if interpolation_type.lower() == "linear":
                    final_embeddings = interpolation.linear_interpolation()
                elif interpolation_type.lower() == "nonlinear":
                    swirl = swirl_factor if swirl_factor is not None else 0.3
                    final_embeddings = interpolation.nonlinear_interpolation(swirl, self.name)
                elif interpolation_type.lower() == "spherical":
                    smooth = smoothness if smoothness is not None else 0.8
                    final_embeddings = interpolation.spherical_interpolation(self.name,smooth)
                else:
                    raise ValueError(f"Unsupported interpolation_type: {interpolation_type}")
            else: final_embeddings = prompt_embeddings
            seed_model = seed if seed is not None else 438233955
            generator = torch.Generator(device=self.DEVICE).manual_seed(seed_model)
            result = self.pipe(
                prompt_embeds=final_embeddings,
                num_inference_steps=30,
                height=512,
                width=512,
                generator=generator
            )
            return result.images[0]

    def _get_embeddings(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).input_ids.to(self.DEVICE)
        with torch.no_grad():
            embeddings = self.text_encoder(tokens).last_hidden_state
        return embeddings