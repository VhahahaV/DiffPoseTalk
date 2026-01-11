from transformers import HubertModel
from transformers.modeling_outputs import BaseModelOutput

from .wav2vec2 import linear_interpolation

_CONFIG_FOR_DOC = 'HubertConfig'


class HubertModel(HubertModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_values, output_fps=25, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, frame_num=None):
        # Temporarily set attention implementation to 'eager' if needed
        original_attn_impl = getattr(self.config, 'attn_implementation', None)
        if hasattr(self.config, 'attn_implementation') and self.config.attn_implementation == 'sdpa':
            self.config.attn_implementation = 'eager'

        # Try to set output_attentions, but skip if it fails due to attn_implementation conflict
        try:
            self.config.output_attentions = True
        except ValueError:
            # If setting output_attentions fails due to attn_implementation, skip it
            pass

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)  # (N, C, L)
        # Resample the audio feature @ 50 fps to `output_fps`.
        if frame_num is not None:
            extract_features_len = round(frame_num * 50 / output_fps)
            extract_features = extract_features[:, :, :extract_features_len]
        extract_features = linear_interpolation(extract_features, 50, output_fps, output_len=frame_num)
        extract_features = extract_features.transpose(1, 2)  # (N, L, C)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # Restore original attention implementation
        if original_attn_impl is not None:
            self.config.attn_implementation = original_attn_impl

        # Restore original attention implementation
        if original_attn_impl is not None:
            self.config.attn_implementation = original_attn_impl

        # Restore original attention implementation
        if original_attn_impl is not None:
            self.config.attn_implementation = original_attn_impl

        try:
            self.config.output_attentions = False
        except ValueError:
            # If setting output_attentions fails, skip it
            pass

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_outputs.hidden_states,
                               attentions=encoder_outputs.attentions, )
