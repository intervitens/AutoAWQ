from .base import BaseAWQForCausalLM
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import MolmoForConditionalGeneration
    from transformers.models.molmo.modeling_molmo import MolmoTextDecoderLayer

class MolmoAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MolmoTextDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model: "MolmoForConditionalGeneration"):
        return model.language_model.model.layers

    @staticmethod
    def get_act_for_scaling(module: "MolmoForConditionalGeneration"):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: "MolmoForConditionalGeneration", device: str):
        model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.to(device)
        #model.vision_tower = model.vision_tower.to(device)
        #model.adapter = model.adapter.to(device)
        model.language_model.model.rotary_emb = model.language_model.model.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module: "MolmoTextDecoderLayer", input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.fc1, module.mlp.activation_fn],
                inp=input_feat["mlp.fc1"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.activation_fn,
                layers=[module.mlp.fc2],
                inp=input_feat["mlp.fc2"],
            )
        )

        return layers
