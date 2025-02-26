import torch
from typing import List

class InterpolationStyles:
    def __init__(self,
                 prompt_embeddings: torch.Tensor,
                 styles_embeddings: List[torch.Tensor],
                 styles_weights: List[float]):
        self.prompt_embeddings: torch.Tensor = prompt_embeddings
        self.styles_embeddings: List[torch.Tensor] = styles_embeddings
        self.styles_weights: List[float] = styles_weights

    def linear_interpolation(self) -> torch.Tensor:
        final_embeddings = (1 - sum(self.styles_weights)) * self.prompt_embeddings
        for style_embedding, style_weight in zip(self.styles_embeddings, self.styles_weights):
            final_embeddings += style_embedding * style_weight
        return final_embeddings

    def nonlinear_interpolation(self, swirl_factor: float, model_name: str) -> torch.Tensor:
        if "Kandinsky" in model_name:
            return self._nonlinear_interpolation_kandinsky(swirl_factor)
        else:
            return self._nonlinear_interpolation(swirl_factor)

    def _nonlinear_interpolation(self, swirl_factor: float) -> torch.Tensor:
        final_embeddings = self.prompt_embeddings.clone()
        for style_embedding, style_weight in zip(self.styles_embeddings, self.styles_weights):
            alpha = torch.sigmoid(torch.linspace(-2, 2, self.prompt_embeddings.shape[1]) * style_weight)
            alpha = alpha.view(1, -1, 1)
            mixed_embeddings = final_embeddings * (1 - alpha) + style_embedding * alpha
            noise = torch.randn_like(mixed_embeddings) * swirl_factor
            final_embeddings = mixed_embeddings + noise * (alpha ** 2)
        return final_embeddings

    def _nonlinear_interpolation_kandinsky(self, swirl_factor: float) -> torch.Tensor:
        final_embeddings = self.prompt_embeddings.clone()
        for style_embedding, style_weight in zip(self.styles_embeddings, self.styles_weights):
            alpha = torch.sigmoid(torch.linspace(-2, 2, self.prompt_embeddings.shape[-1]) * style_weight)
            alpha = alpha.view(1, -1)
            mixed_embeddings = final_embeddings * (1 - alpha) + style_embedding * alpha
            noise = torch.randn_like(mixed_embeddings) * swirl_factor
            final_embeddings = mixed_embeddings + noise * (alpha ** 2)
        return final_embeddings