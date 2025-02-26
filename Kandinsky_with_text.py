import torch
from typing import List, Optional, Union, Tuple
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from Interpolation_styles import InterpolationStyles

class KandinskyWithText:
    MODEL_PATH = "D:/Models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    def __init__(self):
        self.name: str = "Kandinsky 2.2"
        self.pipe_prior = self._initialize_pipeline("kandinsky-community/kandinsky-2-2-prior", prior=True)
        self.pipe_decoder = self._initialize_pipeline("kandinsky-community/kandinsky-2-2-decoder", prior=False)
        print(f"Model {self.name} loaded on {self.DEVICE}")

    def _initialize_pipeline(self,
                             model_name: str,
                             prior: bool) -> Union[KandinskyV22Pipeline, KandinskyV22PriorPipeline]:
        dtype = torch.float16 if self.DEVICE == "cuda" else torch.float32
        try:
            if prior:
                pipe = KandinskyV22PriorPipeline.from_pretrained(
                    model_name,
                    cache_dir=self.MODEL_PATH,
                    torch_dtype=dtype
                ).to(self.DEVICE)
            else:
                pipe = KandinskyV22Pipeline.from_pretrained(
                    model_name,
                    cache_dir=self.MODEL_PATH,
                    torch_dtype=dtype
                ).to(self.DEVICE)
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline {model_name}: {str(e)}")

    def _get_embeddings(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prior_output = self.pipe_prior(
                prompt=prompt,
                negative_prompt="")
        return prior_output.image_embeds, prior_output.negative_image_embeds

    def generate_image(self,
                       prompt:str,
                       style_prompts: Optional[List[str]] = None,
                       style_weights: Optional[List[float]] = None,
                       interpolation_type: str = "linear") -> torch.Tensor:
        if style_prompts is None:
            style_prompts = []
        if style_weights is None:
            style_weights = []

        if len(style_prompts) != len(style_weights):
            raise ValueError("The number of styles must match the number of weights.")
        if not (0 <= sum(style_weights) <= 1):
            raise ValueError("Sum of style weights must be between 0 and 1.")

        with torch.no_grad():
            prompt_embeddings, negative_prompt_embeddings = self._get_embeddings(prompt)
            if style_prompts:
                styles_embeddings = [self._get_embeddings(style_prompt)[0] for style_prompt in style_prompts]
                interpolation = InterpolationStyles(prompt_embeddings, styles_embeddings, style_weights)
                if interpolation_type.lower() == "linear":
                    final_embeddings = interpolation.linear_interpolation()
                else:
                    final_embeddings = interpolation.nonlinear_interpolation(0.3, self.name)
            else: final_embeddings = prompt_embeddings
            generator = torch.Generator(self.DEVICE).manual_seed(438233955)
            image = self.pipe_decoder(
                image_embeds=final_embeddings,
                negative_image_embeds=negative_prompt_embeddings,
                num_inference_steps=50,
                generator=generator,
                height=512,
                width=512
            ).images[0]
        return image