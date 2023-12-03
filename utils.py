import torch

import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from rembg import remove

from torchvision.transforms import CenterCrop, Resize
import numpy as np
import cv2
from typing import Optional
from transformers.modeling_outputs import BaseModelOutputWithPooling

optimizer_dict = {
    "adam": torch.optim.Adam,
    "adamax": torch.optim.Adamax,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adamw": torch.optim.AdamW,
    "sparse_adam": torch.optim.SparseAdam,
    "lbfgs": torch.optim.LBFGS,
    "asgd": torch.optim.ASGD,
    "rprop": torch.optim.Rprop,
    "radam": torch.optim.RAdam,
    "nadam": torch.optim.NAdam,
}

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def cos_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x @ y.T


def make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device,
                     past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def get_last_hidden_state(text_encoder, token_embeddings, position_embeddings, output_hidden_states=True):
    token_embeddings = token_embeddings + position_embeddings
    causal_attention_mask = make_causal_mask((1, token_embeddings.shape[1]), token_embeddings.dtype, token_embeddings.device).to(
        token_embeddings.device)

    output = text_encoder.text_model.encoder(
        inputs_embeds=token_embeddings,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask,
        output_attentions=None,
        output_hidden_states=output_hidden_states,  # could try to align intermeidate states too?
        return_dict=None,
    )

    intermediates = output.hidden_states
    last_hidden_state = text_encoder.text_model.final_layer_norm(output.last_hidden_state)

    normed_intermediates = []
    for i, intermediate in enumerate(intermediates):
        normed_intermediates.append(text_encoder.text_model.final_layer_norm(intermediate))

    return last_hidden_state, normed_intermediates


def get_hidden_states(text_encoder, token_embeddings, position_embeddings, output_hidden_states=True, norm_hidden_states=True):
    token_embeddings = token_embeddings + position_embeddings
    causal_attention_mask = make_causal_mask((1, token_embeddings.shape[1]), token_embeddings.dtype, token_embeddings.device).to(
        token_embeddings.device)

    output = text_encoder.text_model.encoder(
        inputs_embeds=token_embeddings,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask,
        output_attentions=None,
        output_hidden_states=output_hidden_states,  # could try to align intermeidate states too?
        return_dict=None,
    )

    hidden_state = output.hidden_states

    if norm_hidden_states:
        normed_hidden_states = []
        for i, intermediate in enumerate(hidden_state):
            normed_hidden_states.append(text_encoder.text_model.final_layer_norm(intermediate))
        hidden_state = normed_hidden_states

    return hidden_state


def get_text_embed(text_projection, last_hidden_state, first_eof):
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), first_eof]

    text_embedding = text_projection(pooled_output)

    return text_embedding


def get_tokens(tokenizer, prompt, max_length=None):
    if max_length is None:
        max_length = tokenizer.model_max_length

    tokens = tokenizer(prompt, padding="max_length",
                       max_length=max_length,
                       truncation=True,
                       return_tensors="pt").input_ids
    return tokens


def get_similarity(x, y, use_cos_sim=False):
    if use_cos_sim:
        return cos_loss(x, y)
    else:
        return spherical_dist_loss(x, y)


def get_image_embedding(clip_model, hidden_state):
    pooled_output = hidden_state[:, 0, :]
    pooled_output = clip_model.vision_model.post_layernorm(pooled_output)
    image_embedding = clip_model.visual_projection(pooled_output)

    return image_embedding

def get_attn_mask(image):
    image = Resize(224)(image)
    image = CenterCrop(224)(image)
    orig_image_nobg = remove(image)
    mask = np.array(orig_image_nobg)[:,:,-1:]
    mask = np.where(mask > 127.5, 1, 0)
    mask = cv2.resize(mask.squeeze(), (16, 16), interpolation=cv2.INTER_NEAREST)
    attn_mask = torch.from_numpy(mask)
    attn_mask = attn_mask.squeeze().reshape(-1)
    attn_mask = torch.cat([torch.tensor([1]), attn_mask])
    orig_attn_mask = attn_mask.clone()
    attn_mask = attn_mask[None, :].repeat(attn_mask.shape[0], 1)
    attn_mask[~orig_attn_mask.bool()] = 0
    attn_mask[orig_attn_mask.bool()] = orig_attn_mask
    attn_mask = attn_mask.bool()

    return attn_mask, mask

# def optimization_loop(optimizer, token_embeddings, position_embeddings, image_embeds, anchor_embeddings,
#                       use_cos_sim, first_eof, image_factor, anchor_factor, intermediate_similarity, image_intermediates,
#                       anchor_intermediates, intermediate_weights):
#     optimizer.zero_grad()
#
#     last_hidden_state, intermediates = get_last_hidden_state(text_encoder, token_embeddings, position_embeddings)
#     intermediates = [get_text_embed(intermediate, first_eof) for intermediate in intermediates]
#     text_embedding = get_text_embed(last_hidden_state, first_eof)
#
#     image_distance = get_similarity(text_embedding, image_embeds, use_cos_sim=use_cos_sim) * image_factor
#     anchor_distance = get_similarity(text_embedding, anchor_embeddings,
#                                           use_cos_sim=use_cos_sim) * anchor_factor
#
#     if intermediate_similarity:
#         for j in range(len(intermediates)):
#             image_distance += get_similarity(intermediates[j], image_intermediates[j],
#                                                   use_cos_sim=use_cos_sim) * image_factor * intermediate_weights[j]
#             anchor_distance += get_similarity(intermediates[j], anchor_intermediates[j],
#                                                    use_cos_sim=use_cos_sim) * anchor_factor * intermediate_weights[j]
#
#     loss = image_distance + anchor_distance
#     loss.backward()
#     optimizer.step()


def vitg_visual(clip_model_2, x, attn_mask=None):
    # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
    if clip_model_2.input_patchnorm:
        # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
        x = x.reshape(x.shape[0], x.shape[1], clip_model_2.grid_size[0], clip_model_2.patch_size[0],
                      clip_model_2.grid_size[1],
                      clip_model_2.patch_size[1])
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(x.shape[0], clip_model_2.grid_size[0] * clip_model_2.grid_size[1], -1)
        x = clip_model_2.patchnorm_pre_ln(x)
        x = clip_model_2.conv1(x)
    else:
        x = clip_model_2.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # class embeddings and positional embeddings
    x = torch.cat(
        [clip_model_2.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                     device=x.device),
         x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model_2.positional_embedding.to(x.dtype)

    # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
    x = clip_model_2.patch_dropout(x)
    x = clip_model_2.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    intermediates = []
    for block in clip_model_2.transformer.resblocks:
        x = block(x, attn_mask=attn_mask)
        intermediates.append(x)
    intermediates = [x.permute(1, 0, 2) for x in intermediates]
    intermediates = [clip_model_2._global_pool(x)[0] for x in intermediates]
    intermediates = [clip_model_2.ln_post(x) for x in intermediates]
    intermediates = [x @ clip_model_2.proj for x in intermediates]

    final = intermediates.pop(-1)

    return final, intermediates

def get_position_embeddings(text_encoder, num_tokens):
    position_embeddings = text_encoder.text_model.embeddings.position_embedding(text_encoder.text_model.embeddings.position_ids).requires_grad_(False).to(torch.float16)

    # if more than 77 tokens, interpolate
    if num_tokens > 77:
        position_embeddings = F.interpolate(position_embeddings.permute(0,2,1),
                                            size=(num_tokens), mode='linear', align_corners=False).permute(0,2,1)

    return position_embeddings


def text_encoder_forward(text_encoder, input_ids, position_embeddings, attention_mask=None):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = text_encoder.embeddings.token_embedding(input_ids)
        hidden_states = inputs_embeds + position_embeddings

        causal_attention_mask = make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)

        encoder_outputs = text_encoder.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_encoder.final_layer_norm(last_hidden_state)

        output_dict = {
            "last_hidden_state": last_hidden_state,
            "hidden_states": encoder_outputs[1],
        }

        return output_dict


def clip_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.vision_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.vision_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.vision_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask=attention_mask
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



def optimize_chunk(tokenizer,
                   text_encoder,
                   clip_model,
                   feature_extractor,
                   text_projection,
                   prompt,
                   negative_prompt,
                   batch_size,
                   image_prompt,
                   steps=10,
                   lr=0.01,
                   image_factor=1.0,
                   anchor_factor=1.0,
                   intermediate_similarity=False,
                   intermediate_weight_pow=1.5,
                   optimizer="adam",
                   random_init=False,
                   actual_eof=True,
                   intermediate_manual_weights=None,
                   num_tokens_chunk=77,
                   lr_monitor=True,
                   lr_penalty=0.915,
                   clip_num=1,
                   modality_gap_embed=None,
                   isolate_subject=False,
                   image_weights=None,
                   retain_orig_prompt=False,
                   ema_decay=0.0,
                   ):

    text_encoder.requires_grad_(False)

    # we will take a weighted sum of all the produced image embeddings if more than 1 image is provided
    if image_weights is None:
        image_weights = [1 for _ in range(len(image_prompt))]
    image_weights = torch.tensor([weight/sum(image_weights) for weight in image_weights]).cuda()[:,None]

    # if enabled, we will optimize our prompt embeds while its concatenated to the original prompt,
    # just another thing to try
    if retain_orig_prompt:
        num_tokens_chunk = num_tokens_chunk + 77

    with torch.no_grad():
        # whether to also optimize a negative prompt
        use_negative_prompt = negative_prompt is not None

        # clip vitl and clip vitg have a diff number of intermediate states
        intermediate_weights = torch.linspace(0, 1, 13) if clip_num == 1 else torch.linspace(0, 1, 6)
        intermediate_weights = torch.pow(intermediate_weights, intermediate_weight_pow).tolist()

        num_intermediates = len(intermediate_weights)
        intermediate_manual_weights = intermediate_weights

        tokens = get_tokens(tokenizer, prompt, max_length=num_tokens_chunk).to("cuda")
        orig_tokens = get_tokens(tokenizer, prompt).to("cuda")

        # CLIP normally decides for the first instance of an EOF token to be the text embedding pooled output, for the purpose of optimizing
        # our whole set of tokens, we force it to be the last token
        first_eof = torch.argmax(tokens, dim=-1) if not actual_eof else tokens.shape[-1] - 1
        first_real_eof = torch.argmax(orig_tokens, dim=-1)
        if retain_orig_prompt:
            first_eof = first_eof + 77

        position_embeddings = get_position_embeddings(text_encoder,
                                                      num_tokens_chunk
                                                      )
        anchor_pos_embeds = get_position_embeddings(text_encoder,
                                                    77)

        anchor_tokens = get_tokens(tokenizer, prompt).to("cuda")
        output = text_encoder_forward(text_encoder.text_model, anchor_tokens, anchor_pos_embeds)
        anchor_intermediates = output["hidden_states"]
        anchor_intermediates = [get_text_embed(text_projection, anchor_intermediate, first_real_eof) for anchor_intermediate in anchor_intermediates]
        anchor_embeddings = get_text_embed(text_projection, output['last_hidden_state'], first_real_eof).requires_grad_(False).to(torch.float16)

        positive_output = text_encoder_forward(text_encoder.text_model, orig_tokens, anchor_pos_embeds)
        positive_intermediates = positive_output['hidden_states']
        positive_intermediates = [get_text_embed(text_projection, intermediate, first_real_eof) for intermediate in positive_intermediates]
        positive_embedding = get_text_embed(text_projection, positive_output['last_hidden_state'], first_real_eof).requires_grad_(False).to(torch.float16)

        token_embeddings_ = text_encoder.text_model.embeddings.token_embedding(tokens)
        orig_prompt_tok_embeds = text_encoder.text_model.embeddings.token_embedding(orig_tokens)
        if random_init:
            mean, std = token_embeddings_.mean(), token_embeddings_.std()
            token_embeddings_ = torch.normal(mean, std, size=token_embeddings_.shape, device="cuda")

        token_embeddings = deepcopy(token_embeddings_).float().requires_grad_(True)

        pixel_values = [feature_extractor(img, return_tensors="pt").pixel_values.to("cuda").to(position_embeddings.dtype) for img in image_prompt]
        attn_masks = [get_attn_mask(img)[0] for img in image_prompt] if isolate_subject else [None] * len(pixel_values)

        image_embeds, image_intermediates = [], []
        for i, pixel_values_ in enumerate(pixel_values):
            mask = attn_masks[i]
            if clip_num == 1:
                if mask is not None:
                    mask = mask.unsqueeze(0).unsqueeze(0).cuda()

                image_output = clip_forward(clip_model,pixel_values=pixel_values_, output_hidden_states=True, attention_mask=mask)
                img_intermediates = image_output.hidden_states
                img_intermediates = [get_image_embedding(clip_model, img_embed) for img_embed in img_intermediates]
                img_intermediates = [img_embed for i, img_embed in enumerate(img_intermediates) if i % 2 == 0]
                embeds = image_output.pooler_output
                embeds = clip_model.visual_projection(embeds)
            else:
                if mask is not None:
                    mask = mask.cuda()
                embeds, img_intermediates = vitg_visual(clip_model, pixel_values_, attn_mask=mask)

            if modality_gap_embed is not None:
                embeds = embeds + modality_gap_embed

            image_embeds.append(embeds)
            image_intermediates.append(img_intermediates)

        #image_embeds = torch.stack(image_embeds).mean(dim=0)
        image_embeds = (torch.cat(image_embeds) * image_weights).sum(0, keepdim=True)

        ##### get initial distance, this will be used for auto lr
        # TODO auto set lr to start
        with torch.no_grad():
            last_hidden_state, _ = get_last_hidden_state(text_encoder,
                                                         token_embeddings_ if not retain_orig_prompt else torch.cat([orig_prompt_tok_embeds, token_embeddings_],dim=1),
                                                         position_embeddings)
            text_embedding = get_text_embed(text_projection, last_hidden_state, first_eof)
            initial_image_distance = (get_similarity(text_embedding, image_embeds))

        if use_negative_prompt:
            negative_tokens = get_tokens(tokenizer, negative_prompt, max_length=num_tokens_chunk).to("cuda")
            orig_neg_tokens = get_tokens(tokenizer, negative_prompt).to("cuda")
            neg_first_eof = torch.argmax(negative_tokens, dim=-1) if not actual_eof else negative_tokens.shape[-1] - 1
            if retain_orig_prompt:
                neg_first_eof = neg_first_eof + 77
            negative_token_embeddings_ = text_encoder.text_model.embeddings.token_embedding(negative_tokens)
            neg_orig_prompt_tok_embeds = text_encoder.text_model.embeddings.token_embedding(orig_neg_tokens)
            if random_init:
                mean, std = negative_token_embeddings_.mean(), negative_token_embeddings_.std()
                negative_token_embeddings_ = torch.normal(mean, std, size=negative_token_embeddings_.shape, device="cuda")

            negative_token_embeddings = negative_token_embeddings_.float().requires_grad_(True)

        # set up optimizer
        stuff_to_optimize = [token_embeddings]
        if use_negative_prompt:
            stuff_to_optimize.append(negative_token_embeddings)
        optim_class = optimizer_dict[optimizer]
        optimizer = optim_class(stuff_to_optimize, lr=lr)

        if ema_decay > 0.0:
            ema_toks = deepcopy(token_embeddings)
            ema_toks.requires_grad_(False)
            if use_negative_prompt:
                ema_neg_toks = deepcopy(negative_token_embeddings)
                ema_neg_toks.requires_grad_(False)

        if intermediate_manual_weights is not None:
            len_intermediates = len(intermediate_manual_weights)
            image_intermediates = image_intermediates[len(image_intermediates) - len_intermediates:]
            positive_intermediates = positive_intermediates[len(positive_intermediates) - len_intermediates:]
            intermediate_weights = intermediate_manual_weights

        last_image_distance = initial_image_distance
        print("initial image distance", initial_image_distance)

    with torch.enable_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in tqdm(range(steps)):
                optimizer.zero_grad()

                last_hidden_state, intermediates = get_last_hidden_state(text_encoder,
                                                                         token_embeddings if not retain_orig_prompt else torch.cat([orig_prompt_tok_embeds, token_embeddings],dim=1),
                                                                         position_embeddings)
                intermediates = [get_text_embed(text_projection, intermediate, first_eof) for intermediate in
                                 intermediates]
                text_embedding = get_text_embed(text_projection, last_hidden_state, first_eof)

                image_distance = get_similarity(text_embedding, image_embeds) * image_factor
                anchor_distance = get_similarity(text_embedding, anchor_embeddings) * anchor_factor
                last_anchor_distance = anchor_distance.clone().detach().item()

                # drop lr if image distance is increasing
                if lr_monitor and image_distance.item() > last_image_distance:
                    print("too high! dropping lr")
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * lr_penalty

                last_image_distance = image_distance.clone().detach().item() * (1 / image_factor)

                if intermediate_similarity:
                    if intermediate_manual_weights is not None:
                        intermediates = intermediates[len(intermediates) - len_intermediates:]
                    for j in range(len(intermediates)):
                        image_distance += get_similarity(intermediates[j], image_intermediates[j]) * intermediate_weights[j] * image_factor
                        anchor_distance += get_similarity(intermediates[j], anchor_intermediates[j]) * intermediate_weights[j] * anchor_factor

                loss = anchor_distance
                loss += image_distance

                if use_negative_prompt:
                    neg_output_embeddings, negative_intermediates = get_last_hidden_state(text_encoder,
                                                                                            negative_token_embeddings if not retain_orig_prompt else torch.cat([neg_orig_prompt_tok_embeds, negative_token_embeddings],dim=1),
                                                                                            position_embeddings)
                    negative_intermediates = [get_text_embed(text_projection, negative_intermediate, neg_first_eof) for negative_intermediate in negative_intermediates]
                    neg_text_embedding = get_text_embed(text_projection, neg_output_embeddings, neg_first_eof)

                    neg_image_distance = get_similarity(neg_text_embedding, image_embeds) * image_factor * -1
                    distance_from_prompt = get_similarity(neg_text_embedding, positive_embedding) * 0  # * -1

                    if intermediate_similarity:
                        if intermediate_manual_weights is not None:
                            negative_intermediates = negative_intermediates[len(negative_intermediates) - len_intermediates:]
                        for j in range(len(negative_intermediates)):
                            neg_image_distance += get_similarity(negative_intermediates[j],image_intermediates[j]) * -1 * intermediate_weights[j]
                            distance_from_prompt += get_similarity(negative_intermediates[j],
                                                                        positive_intermediates[j]) * -1 * \
                                                    intermediate_weights[j]

                    loss = loss + distance_from_prompt
                    loss = loss + neg_image_distance


                loss.backward()
                optimizer.step()

                if ema_decay > 0.0:
                    with torch.no_grad():
                        ema_toks = ema_decay * token_embeddings + (1 - ema_decay) * ema_toks
                        token_embeddings = ema_toks
                        if use_negative_prompt:
                            ema_neg_toks = ema_decay * negative_token_embeddings + (1 - ema_decay) * ema_neg_toks
                            negative_token_embeddings = ema_neg_toks


    with torch.no_grad():
        print("final image distance", last_image_distance)
        print("final anchor distance", last_anchor_distance)

        token_embeddings = token_embeddings.detach().to(torch.float16)
        position_embeddings = get_position_embeddings(text_encoder, token_embeddings.shape[1])

        if clip_num > 1:
            prompt_embeds = get_hidden_states(text_encoder, token_embeddings, position_embeddings, norm_hidden_states=False)[-2]
        else:
            prompt_embeds = get_hidden_states(text_encoder, token_embeddings, position_embeddings)[-1]
        prompt_embeds.repeat(batch_size, 1, 1)

        if use_negative_prompt:
            negative_token_embeddings = negative_token_embeddings.detach().to(torch.float16)
            if clip_num > 1:
                negative_prompt_embeds = get_hidden_states(text_encoder, negative_token_embeddings, position_embeddings, norm_hidden_states=False)[-2]
            else:
                negative_prompt_embeds = get_hidden_states(text_encoder, negative_token_embeddings, position_embeddings)[-1]
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        if clip_num > 1:
            last_hidden_state, _ = get_last_hidden_state(text_encoder, token_embeddings, position_embeddings)
            pooled = get_text_embed(text_projection, last_hidden_state, first_eof)

            if use_negative_prompt:
                neg_output_embeddings, _ = get_last_hidden_state(text_encoder, negative_token_embeddings, position_embeddings)
                neg_pooled = get_text_embed(text_projection, neg_output_embeddings, neg_first_eof)
            else:
                neg_pooled = torch.zeros_like(pooled)

        else:
            pooled = None
            neg_pooled = None

    return prompt_embeds, negative_prompt_embeds, pooled, neg_pooled


def optimize_prompt(tokenizer,
                    text_encoder,
                    clip_model,
                    feature_extractor,
                    text_projection,
                    prompt,
                    image_prompt,
                    anchor_prompt,
                    batch_size,
                    negative_prompt=None,
                    use_negative_prompt=False,
                    steps=10,
                    lr=0.01,
                    optimizer="adam",
                    use_cos_sim=False,
                    image_factor=1.0,
                    anchor_factor=1.0,
                    intermediate_similarity=False,
                    intermediate_weight_pow=1.5,
                    actual_eof=True,
                    random_init=False,
                    intermediate_manual_weights=None,
                    lr_monitor=True,
                    lr_penalty=0.915,
                    clip_num=1,
                    modality_gap_embed=None,
                    isolate_subject=False,
                    image_weights=None,
                    num_tokens=77,
                    ema_decay=0.0,
                    ):

    if image_weights is None:
        image_weights = [1 for _ in range(len(image_prompt))]

    image_weights = torch.tensor([weight / sum(image_weights) for weight in image_weights]).cuda()[:, None]

    if negative_prompt is None and use_negative_prompt:
        negative_prompt = ""

    with torch.no_grad():
        # image intermediates has 25, 48
        # text intermediates has 13, 32
        text_encoder.requires_grad_(False)

        # clip vitl and clip vitg have a diff number of intermediate states
        intermediate_weights = torch.linspace(0, 1, 13) if clip_num == 1 else torch.linspace(0, 1, 6)
        intermediate_weights = torch.pow(intermediate_weights, intermediate_weight_pow).tolist()

        num_intermediates = len(intermediate_weights)
        intermediate_manual_weights = intermediate_weights

        tokens = get_tokens(tokenizer, prompt, max_length=num_tokens).to("cuda")
        first_eof = torch.argmax(tokens, dim=-1) if actual_eof else tokens.shape[-1] - 1
        first_real_eof = torch.argmax(tokens, dim=-1)

        position_embeddings = get_position_embeddings(text_encoder, num_tokens)

        pixel_values = [feature_extractor(img, return_tensors="pt").pixel_values.to("cuda").to(text_encoder.dtype) for img in image_prompt]
        attn_masks = [get_attn_mask(img)[0] for img in image_prompt] if isolate_subject else [None] * len(pixel_values)

        image_embeds, image_intermediates = [], []
        for i, pixel_values_ in enumerate(pixel_values):
            mask = attn_masks[i]
            if clip_num == 1:
                if mask is not None:
                    mask = mask.unsqueeze(0).unsqueeze(0).cuda()

                image_output = clip_forward(clip_model,pixel_values=pixel_values_, output_hidden_states=True, attention_mask=mask)
                img_intermediates = image_output.hidden_states
                img_intermediates = [get_image_embedding(clip_model, img_embed) for img_embed in img_intermediates]
                img_intermediates = [img_embed for i, img_embed in enumerate(img_intermediates) if i % 2 == 0]
                embeds = image_output.pooler_output
                embeds = clip_model.visual_projection(embeds)
            else:
                if mask is not None:
                    mask = mask.cuda()
                embeds, img_intermediates = vitg_visual(clip_model, pixel_values_, attn_mask=mask)

            if modality_gap_embed is not None:
                embeds = embeds + modality_gap_embed

            image_embeds.append(embeds)
            image_intermediates.append(img_intermediates)

        #image_embeds = torch.stack(image_embeds).mean(dim=0)
        image_embeds = (torch.cat(image_embeds) * image_weights).sum(0, keepdim=True)

        anchor_tokens = get_tokens(tokenizer, anchor_prompt).to("cuda")
        output = text_encoder(anchor_tokens, output_hidden_states=True)
        anchor_intermediates = output.hidden_states
        anchor_intermediates = [get_text_embed(text_projection, anchor_intermediate, first_real_eof) for
                                anchor_intermediate in anchor_intermediates]
        anchor_embeddings = get_text_embed(text_projection, output.last_hidden_state, first_real_eof).requires_grad_(
            False).to(torch.float16)


        token_embeddings_ = text_encoder.text_model.embeddings.token_embedding(tokens)
        if random_init:
            mean, std = token_embeddings_.mean(), token_embeddings_.std()
            token_embeddings_ = torch.normal(mean, std, size=token_embeddings_.shape, device="cuda")
        token_embeddings = deepcopy(token_embeddings_).float().requires_grad_(True)

        ##### get initial distance, this will be used for auto lr
        # TODO auto set lr to start
        with torch.no_grad():
            last_hidden_state, _ = get_last_hidden_state(text_encoder,
                                                         token_embeddings_,
                                                         position_embeddings)
            text_embedding = get_text_embed(text_projection, last_hidden_state, first_eof)
            initial_image_distance = get_similarity(text_embedding, image_embeds)

        if use_negative_prompt:
            negative_tokens = get_tokens(tokenizer, negative_prompt, max_length=num_tokens).to("cuda")
            if actual_eof:
                neg_first_eof = torch.argmax(negative_tokens, dim=-1)
            else:
                neg_first_eof = negative_tokens.shape[-1] - 1
            neg_first_real_eof = torch.argmax(negative_tokens, dim=-1)

            negative_token_embeddings_ = text_encoder.text_model.embeddings.token_embedding(negative_tokens)
            if random_init:
                mean, std = negative_token_embeddings_.mean(), negative_token_embeddings_.std()
                negative_token_embeddings_ = torch.normal(mean, std, size=negative_token_embeddings_.shape, device="cuda")

            negative_anchor_tokens = get_tokens(tokenizer, negative_prompt).to("cuda")
            neg_output = text_encoder(negative_anchor_tokens, output_hidden_states=True)
            negative_anchor_intermediates = neg_output.hidden_states
            negative_anchor_intermediates = [
                get_text_embed(text_projection, negative_anchor_intermediate, neg_first_real_eof) for
                negative_anchor_intermediate in negative_anchor_intermediates]
            negative_anchor_embeddings = get_text_embed(text_projection, neg_output.last_hidden_state, neg_first_real_eof).requires_grad_(False).to(torch.float16)
            negative_token_embeddings = negative_token_embeddings_.float().requires_grad_(True)

        optim_class = optimizer_dict[optimizer]
        stuff_to_optimize = [token_embeddings]
        if use_negative_prompt:
            stuff_to_optimize.append(negative_token_embeddings)
        optimizer = optim_class(stuff_to_optimize, lr=lr)

        if ema_decay > 0.0:
            ema_toks = deepcopy(token_embeddings)
            ema_toks.requires_grad_(False)
            if use_negative_prompt:
                ema_neg_toks = deepcopy(negative_token_embeddings)
                ema_neg_toks.requires_grad_(False)

        if intermediate_manual_weights is not None:
            len_intermediates = len(intermediate_manual_weights)
            image_intermediates = image_intermediates[len(image_intermediates) - len_intermediates:]
            anchor_intermediates = anchor_intermediates[len(anchor_intermediates) - len_intermediates:]
            intermediate_weights = intermediate_manual_weights
            if use_negative_prompt:
                negative_anchor_intermediates = negative_anchor_intermediates[
                                                len(negative_anchor_intermediates) - len_intermediates:]

        # intermediate_weights[intermediate_weights<0.03] = 0
        last_image_distance = initial_image_distance
        print("initial image distance", initial_image_distance)

    eps = 1e-6
    with torch.enable_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in tqdm(range(steps)):
                optimizer.zero_grad()

                last_hidden_state, intermediates = get_last_hidden_state(text_encoder, token_embeddings, position_embeddings)
                intermediates = [get_text_embed(text_projection, intermediate, first_eof) for intermediate in
                                 intermediates]
                text_embedding = get_text_embed(text_projection, last_hidden_state, first_eof)

                image_distance = get_similarity(text_embedding, image_embeds, use_cos_sim=use_cos_sim) * image_factor
                anchor_distance = get_similarity(text_embedding, anchor_embeddings, use_cos_sim=use_cos_sim)
                last_anchor_distance = anchor_distance.clone().detach().item()
                anchor_distance = anchor_distance * anchor_factor

                # drop lr if image distance is increasing
                if lr_monitor and image_distance.item() > last_image_distance:
                    print("too high! dropping lr")
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * lr_penalty

                last_image_distance = image_distance.clone().detach().item() * (1 / (image_factor + eps))

                if intermediate_similarity:
                    if intermediate_manual_weights is not None:
                        intermediates = intermediates[len(intermediates) - len_intermediates:]
                    for j in range(len(intermediates)):
                        image_distance += get_similarity(intermediates[j], image_intermediates[j],
                                                                 use_cos_sim=use_cos_sim) * image_factor * \
                                                   intermediate_weights[j]
                        anchor_distance += get_similarity(intermediates[j], anchor_intermediates[j],
                                                               use_cos_sim=use_cos_sim) * anchor_factor * \
                                           intermediate_weights[j]

                loss = anchor_distance + image_distance

                if use_negative_prompt:
                    neg_output_embeddings, negative_intermediates = get_last_hidden_state(text_encoder,
                        negative_token_embeddings, position_embeddings)
                    negative_intermediates = [
                        get_text_embed(text_projection, negative_intermediate, neg_first_eof) for
                        negative_intermediate in negative_intermediates]
                    neg_text_embedding = get_text_embed(text_projection, neg_output_embeddings, neg_first_eof)

                    neg_image_distance = (get_similarity(neg_text_embedding, image_embeds, use_cos_sim=use_cos_sim) * image_factor * -1)
                    neg_anchor_distance = get_similarity(neg_text_embedding, anchor_embeddings,
                                                              use_cos_sim=use_cos_sim) * anchor_factor * -1

                    if intermediate_similarity:
                        if intermediate_manual_weights is not None:
                            negative_intermediates = negative_intermediates[len(negative_intermediates) - len_intermediates:]
                        for j in range(len(negative_intermediates)):
                            neg_image_distance += get_similarity(negative_intermediates[j],
                                                                          image_intermediates[i][j],
                                                                          use_cos_sim=use_cos_sim) * image_factor * -1 * \
                                                          intermediate_weights[j]
                            neg_anchor_distance += get_similarity(negative_intermediates[j],
                                                                       anchor_intermediates[j],
                                                                       use_cos_sim=use_cos_sim) * anchor_factor * \
                                                   intermediate_weights[j] * -1

                    loss = loss + neg_anchor_distance + neg_image_distance

                loss.backward()
                optimizer.step()

                if ema_decay > 0.0:
                    with torch.no_grad():
                        ema_toks = ema_decay * token_embeddings + (1 - ema_decay) * ema_toks
                        token_embeddings = ema_toks
                        if use_negative_prompt:
                            ema_neg_toks = ema_decay * negative_token_embeddings + (1 - ema_decay) * ema_neg_toks
                            negative_token_embeddings = ema_neg_toks


    with torch.no_grad():
        print("final image distance", last_image_distance)
        print("final anchor distance", last_anchor_distance)

        token_embeddings = token_embeddings.detach().to(torch.float16)
        position_embeddings = position_embeddings.detach().to(torch.float16)

        if clip_num > 1:
            prompt_embeds = get_hidden_states(text_encoder, token_embeddings, position_embeddings, norm_hidden_states=False)[-2]
        else:
            prompt_embeds = get_hidden_states(text_encoder, token_embeddings, position_embeddings)[-1]
        prompt_embeds.repeat(batch_size, 1, 1)

        if use_negative_prompt:
            negative_token_embeddings = negative_token_embeddings.detach().to(torch.float16)
            if clip_num > 1:
                negative_prompt_embeds = get_hidden_states(text_encoder, negative_token_embeddings, position_embeddings, norm_hidden_states=False)[-2]
            else:
                negative_prompt_embeds = get_hidden_states(text_encoder, negative_token_embeddings, position_embeddings)[-1]
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        if clip_num > 1:
            last_hidden_state, _ = get_last_hidden_state(text_encoder, token_embeddings, position_embeddings)
            pooled = get_text_embed(text_projection, last_hidden_state, first_eof)

            if use_negative_prompt:
                neg_output_embeddings, _ = get_last_hidden_state(text_encoder, negative_token_embeddings, position_embeddings)
                neg_pooled = get_text_embed(text_projection, neg_output_embeddings, neg_first_eof)
            else:
                neg_pooled = torch.zeros_like(pooled)

        else:
            pooled = None
            neg_pooled = None

    return prompt_embeds, negative_prompt_embeds, pooled, neg_pooled




def optimize_prompt_simple(tokenizer,
                    text_encoder,
                    clip_model,
                    feature_extractor,
                    text_projection,
                    prompt,
                    image_prompt,
                    anchor_prompt,
                    batch_size,
                    negative_prompt=None,
                    use_negative_prompt=False,
                    steps=10,
                    lr=0.01,
                    optimizer="adam",
                    use_cos_sim=False,
                    image_factor=1.0,
                    anchor_factor=1.0,
                    lr_monitor=True,
                    lr_penalty=0.915,
                    clip_num=1,
                    image_weights=None,
                    num_tokens=77,
                    ):

    if image_weights is None:
        image_weights = [1 for _ in range(len(image_prompt))]

    image_weights = torch.tensor([weight / sum(image_weights) for weight in image_weights]).cuda()[:, None]

    if negative_prompt is None and use_negative_prompt:
        negative_prompt = ""

    with torch.no_grad():
        # image intermediates has 25, 48
        # text intermediates has 13, 32
        text_encoder.requires_grad_(False)

        tokens = get_tokens(tokenizer, prompt, max_length=num_tokens).to("cuda")
        first_eof = torch.argmax(tokens, dim=-1)
        first_real_eof = torch.argmax(tokens, dim=-1)

        position_embeddings = get_position_embeddings(text_encoder, num_tokens)

        pixel_values = [feature_extractor(img, return_tensors="pt").pixel_values.to("cuda").to(text_encoder.dtype) for img in image_prompt]

        image_embeds, image_intermediates = [], []
        for i, pixel_values_ in enumerate(pixel_values):
            image_embeds.append(clip_model(pixel_values_).image_embeds)

        #image_embeds = torch.stack(image_embeds).mean(dim=0)
        image_embeds = (torch.cat(image_embeds) * image_weights).sum(0, keepdim=True)

        anchor_tokens = get_tokens(tokenizer, anchor_prompt).to("cuda")
        output = text_encoder(anchor_tokens, output_hidden_states=True)
        anchor_embeddings = get_text_embed(text_projection, output.last_hidden_state, first_real_eof).requires_grad_(False).to(torch.float16)


        token_embeddings_ = text_encoder.text_model.embeddings.token_embedding(tokens)
        token_embeddings = deepcopy(token_embeddings_).float().requires_grad_(True)

        ##### get initial distance, this will be used for auto lr
        # TODO auto set lr to start
        with torch.no_grad():
            last_hidden_state, _ = get_last_hidden_state(text_encoder,
                                                         token_embeddings_,
                                                         position_embeddings)
            text_embedding = get_text_embed(text_projection, last_hidden_state, first_eof)
            initial_image_distance = get_similarity(text_embedding, image_embeds)

        if use_negative_prompt:
            negative_tokens = get_tokens(tokenizer, negative_prompt, max_length=num_tokens).to("cuda")
            neg_first_eof = torch.argmax(negative_tokens, dim=-1)
            negative_token_embeddings_ = text_encoder.text_model.embeddings.token_embedding(negative_tokens)
            negative_token_embeddings = negative_token_embeddings_.float().requires_grad_(True)

        optim_class = optimizer_dict[optimizer]
        stuff_to_optimize = [token_embeddings]
        if use_negative_prompt:
            stuff_to_optimize.append(negative_token_embeddings)
        optimizer = optim_class(stuff_to_optimize, lr=lr)

        # intermediate_weights[intermediate_weights<0.03] = 0
        last_image_distance = initial_image_distance
        print("initial image distance", initial_image_distance)

    eps = 1e-6
    with torch.enable_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in tqdm(range(steps)):
                optimizer.zero_grad()

                last_hidden_state, intermediates = get_last_hidden_state(text_encoder, token_embeddings, position_embeddings)
                text_embedding = get_text_embed(text_projection, last_hidden_state, first_eof)

                image_distance = get_similarity(text_embedding, image_embeds, use_cos_sim=use_cos_sim) * image_factor
                anchor_distance = get_similarity(text_embedding, anchor_embeddings, use_cos_sim=use_cos_sim)
                last_anchor_distance = anchor_distance.clone().detach().item()
                anchor_distance = anchor_distance * anchor_factor

                # drop lr if image distance is increasing
                if lr_monitor and image_distance.item() > last_image_distance:
                    print("too high! dropping lr")
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * lr_penalty

                last_image_distance = image_distance.clone().detach().item() * (1 / (image_factor + eps))

                loss = anchor_distance + image_distance

                if use_negative_prompt:
                    neg_output_embeddings, negative_intermediates = get_last_hidden_state(text_encoder,
                        negative_token_embeddings, position_embeddings)
                    neg_text_embedding = get_text_embed(text_projection, neg_output_embeddings, neg_first_eof)

                    neg_image_distance = (get_similarity(neg_text_embedding, image_embeds, use_cos_sim=use_cos_sim) * image_factor * -1)
                    neg_anchor_distance = get_similarity(neg_text_embedding, anchor_embeddings,
                                                              use_cos_sim=use_cos_sim) * anchor_factor * -1

                    loss = loss + neg_anchor_distance + neg_image_distance

                loss.backward()
                optimizer.step()


    with torch.no_grad():
        print("final image distance", last_image_distance)
        print("final anchor distance", last_anchor_distance)

        token_embeddings = token_embeddings.detach().to(torch.float16)
        position_embeddings = position_embeddings.detach().to(torch.float16)

        prompt_embeds = get_hidden_states(text_encoder, token_embeddings, position_embeddings)[-1]

        if use_negative_prompt:
            negative_token_embeddings = negative_token_embeddings.detach().to(torch.float16)
            negative_prompt_embeds = get_hidden_states(text_encoder, negative_token_embeddings, position_embeddings)[-1]
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    return prompt_embeds, negative_prompt_embeds
