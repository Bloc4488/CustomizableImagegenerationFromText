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
        device = self.prompt_embeddings.device
        final_embeddings = self.prompt_embeddings.clone()
        for style_embedding, style_weight in zip(self.styles_embeddings, self.styles_weights):
            alpha = torch.sigmoid(
                torch.linspace(-2, 2, self.prompt_embeddings.shape[1], device=device) * style_weight)
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

    def spherical_interpolation(self, model_name: str, smoothness: float = 1.0) -> torch.Tensor:
        if "Kandinsky" in model_name:
            return self._spherical_interpolation_Kandinsky(smoothness)
        else:
            return self._spherical_interpolation(smoothness)

    def _spherical_interpolation(self, smoothness: float = 1.0) -> torch.Tensor:
        final_embeddings = self.prompt_embeddings.clone()

        for style_embedding, style_weight in zip(self.styles_embeddings, self.styles_weights):
            t = min(max(style_weight * smoothness, 0.0), 1.0)
            prompt_flat = final_embeddings.view(-1, final_embeddings.shape[-1])
            style_flat = style_embedding.view(-1, style_embedding.shape[-1])
            dot_product = torch.sum(prompt_flat * style_flat, dim=-1, keepdim=True)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            omega = torch.acos(dot_product)

            sin_omega = torch.sin(omega)
            mask = sin_omega > 1e-6
            coeff1 = torch.sin((1.0 - t) * omega) / sin_omega
            coeff2 = torch.sin(t * omega) / sin_omega

            coeff1 = torch.where(mask, coeff1, 1.0 - t)
            coeff2 = torch.where(mask, coeff2, t)

            coeff1 = coeff1.unsqueeze(0).expand_as(final_embeddings)
            coeff2 = coeff2.unsqueeze(0).expand_as(style_embedding)
            final_embeddings = coeff1 * final_embeddings + coeff2 * style_embedding

        return final_embeddings

    def _spherical_interpolation_Kandinsky(self, smoothness: float = 1.0) -> torch.Tensor:
        final_embeddings = self.prompt_embeddings.clone()


        for style_embedding, style_weight in zip(self.styles_embeddings, self.styles_weights):
            t = min(max(style_weight * smoothness, 0.0), 1.0)

            prompt_flat = final_embeddings
            style_flat = style_embedding

            dot_product = torch.sum(prompt_flat * style_flat, dim=-1, keepdim=True)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            omega = torch.acos(dot_product)

            sin_omega = torch.sin(omega)
            mask = sin_omega > 1e-6

            coeff1 = torch.sin((1.0 - t) * omega) / sin_omega
            coeff2 = torch.sin(t * omega) / sin_omega

            coeff1 = torch.where(mask, coeff1, 1.0 - t)
            coeff2 = torch.where(mask, coeff2, t)

            coeff1 = coeff1.expand_as(final_embeddings)
            coeff2 = coeff2.expand_as(style_embedding)
            final_embeddings = coeff1 * final_embeddings + coeff2 * style_embedding

        return final_embeddings