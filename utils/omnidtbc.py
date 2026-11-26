import torch
from contextlib import nullcontext
from typing import Tuple


def profile_section(profile_recorder, name: str):
    if profile_recorder is None:
        return nullcontext()
    return profile_recorder.record(name)

def allocate_tokens_to_windows(scores: torch.Tensor, total_keep: int) -> torch.Tensor:
    B, N = scores.shape
    
    window_scores = scores.sum(dim=-1)
    T = 2.0 
    p = torch.softmax(window_scores / T, dim=-1)
    exact_k = p * total_keep
    
    base_k = exact_k.floor().long()
    base_k = torch.clamp(base_k, min=1, max=N-1)
    
    remainder = total_keep - base_k.sum().item()
    
    while remainder > 0:
        frac = exact_k - base_k.float()
        frac[base_k >= N] = -1.0 
        if frac.max() < 0: break     
        idx = frac.argmax()
        base_k[idx] += 1
        remainder -= 1
        
    while remainder < 0:
        frac = base_k.float() - exact_k
        frac[base_k <= 1] = -1.0 
        if frac.max() < 0: break
        idx = frac.argmax()
        base_k[idx] -= 1
        remainder += 1
        
    return base_k


def batched_greedy_variable_k_dpp(features: torch.Tensor, scores: torch.Tensor, K_array: torch.Tensor) -> torch.Tensor:

    B, N, D = features.shape
    k_max = int(K_array.max().item())
    
    if k_max <= 0:
        return torch.zeros(B, N, dtype=torch.bool, device=features.device)
        
    q = torch.clamp(scores, min=1e-6, max=1.0)
    
    f = torch.nn.functional.normalize(features, p=2, dim=-1) * q.unsqueeze(-1)
    dists = f.norm(dim=-1) ** 2
    
    keep_mask = torch.zeros(B, N, dtype=torch.bool, device=features.device)
    basis = torch.zeros(B, k_max, D, dtype=features.dtype, device=features.device)
    batch_idx = torch.arange(B, device=features.device)

    for step in range(k_max):
        active_mask = (step < K_array)
        
        masked_dists = dists.masked_fill(keep_mask, -1.0)
        idx = masked_dists.argmax(dim=-1)
        
        keep_mask[batch_idx[active_mask], idx[active_mask]] = True
        
        v_new = f[batch_idx, idx] 
        v_new = v_new * active_mask.unsqueeze(-1)
        
        if step > 0:
            prev_basis = basis[:, :step, :]
            proj_coeff = torch.bmm(prev_basis, v_new.unsqueeze(-1))
            proj_vec = torch.bmm(prev_basis.transpose(1, 2), proj_coeff).squeeze(-1)
            v_new = v_new - proj_vec
            
        v_new_norm = v_new.norm(dim=-1, keepdim=True)
        e_new = v_new / (v_new_norm + 1e-8)
        basis[:, step, :] = e_new
        
        proj = torch.bmm(f, e_new.unsqueeze(-1)).squeeze(-1) ** 2
        proj = proj * active_mask.unsqueeze(-1)
        
        dists.sub_(proj).clamp_(min=0.0)
        
    return keep_mask


def merge(
    features: torch.Tensor,
    scores: torch.Tensor,
    keep_mask: torch.Tensor,
    top_ratio_to_merge: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, D = features.shape
    out_features = features.clone()
    out_mask = keep_mask.clone()

    num_unp_all = (~out_mask).sum(dim=1)

    for b in range(B):
        num_unp = num_unp_all[b].item()
        if num_unp == 0:
            continue

        k_to_merge = max(1, int(num_unp * top_ratio_to_merge))
        
        masked_scores = scores[b].masked_fill(out_mask[b], -float('inf'))
        
        _, selected_unp_idx = torch.topk(masked_scores, k_to_merge)

        host_idx = selected_unp_idx[0]

        merged_feat = features[b, selected_unp_idx].mean(dim=0)

        out_features[b, host_idx] = merged_feat
        out_mask[b, host_idx] = True

    return out_features, out_mask


def windowed_dynamic_dpp(
    features: torch.Tensor,
    scores: torch.Tensor,
    keep_ratio: float,
    chunk_size: int,
    profile_recorder=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, D = features.shape
    total_keep = int(N * keep_ratio)
    chunk_size = max(1, int(chunk_size))
    
    if total_keep >= N:
        return features.clone(), torch.ones(N, dtype=torch.bool, device=features.device)
    
    num_chunks = N // chunk_size
    remainder = N % chunk_size
    
    keep_mask = torch.zeros(N, dtype=torch.bool, device=features.device)
    out_features = features.clone() 
    
    MERGE_UNP_RATIO = 0.1 
    
    if num_chunks > 0:
        main_N = num_chunks * chunk_size
        feat_main = features[:main_N].view(num_chunks, chunk_size, D)
        score_main = scores[:main_N].view(num_chunks, chunk_size)
        
        target_keep_main = int(main_N * keep_ratio)
        with profile_section(profile_recorder, "dynamic_token_allocation"):
            K_array = allocate_tokens_to_windows(score_main, target_keep_main)
        with profile_section(profile_recorder, "subset_selection"):
            mask_main = batched_greedy_variable_k_dpp(feat_main, score_main, K_array)
        
        feat_main_out, mask_main_out = merge(
            features=feat_main, 
            scores=score_main, 
            keep_mask=mask_main, 
            top_ratio_to_merge=MERGE_UNP_RATIO
        )

        keep_mask[:main_N] = mask_main.view(-1)
        out_features[:main_N] = feat_main.view(main_N, D)
        
    if remainder > 0:
        feat_rem = features[num_chunks * chunk_size:].unsqueeze(0)
        score_rem = scores[num_chunks * chunk_size:].unsqueeze(0)
        
        with profile_section(profile_recorder, "dynamic_token_allocation"):
            rem_k = torch.tensor([int(remainder * keep_ratio)], device=features.device)
            rem_k = torch.clamp(rem_k, min=1, max=remainder)
        
        with profile_section(profile_recorder, "subset_selection"):
            mask_rem = batched_greedy_variable_k_dpp(feat_rem, score_rem, rem_k)
        feat_rem_out, mask_rem_out = merge(
            feat_rem, score_rem, mask_rem, top_ratio_to_merge=MERGE_UNP_RATIO
        )
        keep_mask[num_chunks * chunk_size:] = mask_rem.squeeze(0)
        out_features[num_chunks * chunk_size:] = feat_rem.squeeze(0)
        
    return out_features, keep_mask

def min_max_norm(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s_min = scores.min()
    s_max = scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min + eps)
    else:
        return torch.ones_like(scores)

def omnidtbc(
    input_embeds: torch.Tensor,
    audio_attn_logits: torch.Tensor,
    video_attn_logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    remain_ratio_audio: float = 0.35,
    remain_ratio_video: float = 0.35,
    video_chunk_frames: int = 4,
    alpha: float = 0.1,
    profile_recorder=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    device = input_embeds.device
    
    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids

    video_token_mask = (flat_ids == video_token_id).to(device)
    audio_token_mask = (flat_ids == audio_token_id).to(device)

    video_indices = torch.nonzero(video_token_mask, as_tuple=True)[0]
    audio_indices = torch.nonzero(audio_token_mask, as_tuple=True)[0]

    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]

    video_score = video_attn_logits.reshape(-1)
    audio_score = audio_attn_logits.reshape(-1)
    
    if video_feature.numel() > 0 and audio_feature.numel() > 0:

        import torch.nn.functional as F
        with profile_section(profile_recorder, "cross_modal_similarity_computation"):
            v_norm = F.normalize(video_feature, p=2, dim=-1, eps=1e-6)
            a_norm = F.normalize(audio_feature, p=2, dim=-1, eps=1e-6)
            similarity = a_norm @ v_norm.t()
            audio_score_cross = min_max_norm(similarity.max(dim=-1).values)
            video_score_cross = min_max_norm(similarity.max(dim=0).values)

            video_score_self = min_max_norm(video_attn_logits.reshape(-1))
            audio_score_self = min_max_norm(audio_attn_logits.reshape(-1))

            video_score = video_score_self + alpha * video_score_cross
            audio_score = audio_score_self + alpha * audio_score_cross

    if video_feature.numel() > 0:
        video_tokens_per_frame = video_feature.shape[0] // max(1, num_input_frames)
        video_chunk_size = video_tokens_per_frame * video_chunk_frames
        
        merged_video_feature, final_video_mask = windowed_dynamic_dpp(
            features=video_feature, 
            scores=video_score, 
            keep_ratio=remain_ratio_video, 
            chunk_size=video_chunk_size,
            profile_recorder=profile_recorder,
        )
    else:
        merged_video_feature = video_feature
        final_video_mask = torch.zeros(0, dtype=torch.bool, device=device)

    if audio_feature.numel() > 0:
        audio_chunk_size = min(200, audio_feature.shape[0])
        
        merged_audio_feature, final_audio_mask = windowed_dynamic_dpp(
            features=audio_feature, 
            scores=audio_score, 
            keep_ratio=remain_ratio_audio, 
            chunk_size=audio_chunk_size,
            profile_recorder=profile_recorder,
        )
    else:
        merged_audio_feature = audio_feature
        final_audio_mask = torch.zeros(0, dtype=torch.bool, device=device)

    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    global_mask[video_indices] = final_video_mask
    global_mask[audio_indices] = final_audio_mask

    input_embeds_out = flat_embeds.clone()
    if video_feature.numel() > 0:
        input_embeds_out[video_indices] = merged_video_feature
    if audio_feature.numel() > 0:
        input_embeds_out[audio_indices] = merged_audio_feature

    if is_batched:
        input_embeds_out = input_embeds_out.reshape(B, L, D)

    return input_embeds_out, global_mask
