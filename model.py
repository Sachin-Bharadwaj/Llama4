import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Llama4TextConfig:
    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192 # Sachin: the hidden_size will be projected upto this number
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128 # derived paramater: hidden_size // num_attention_heads
    max_position_embeddings: int = 4096 * 32 # context size
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 200018
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 500000
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 1 #Sachin: router routes each token to only 1 expert
    num_local_experts: int = 16 # Sachin : ??
    use_qk_norm: bool = True # Sachin : ??
    no_rope_layer_interval: int = 4 # Sachin : ??
    attention_chunk_size: int = 8192 # Sachin : ??
    attn_temperature_tuning: float = 4 # Sachin : ??
    floor_scale: int = 8192 # Sachin : ??
    attn_scale: float = 0.1 # Sachin : ??

@dataclass
class Llama4VisionConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5632
    vision_output_size: int = 7680
    image_size: int = 448
    patch_size: int = 14
    norm_eps: float = 1e-5
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    projector_dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_theta: int = 10000


class Llama4TextExperts(nn.Module):
    def __init__(self, config):
        super(Llama4TextExperts, self).__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2*self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = nn.SiLU()

        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(self.num_experts, -1, self.hidden_size) # (num_experts, bs*seq_len, hidden_size)

        gate_up = torch.bmm(hidden_states, self.gate_up_proj) # (num_experts, bs*seq_len, 2*expert_dim)
        
        gate, up = gate_up.chunk(2, dim=-1) # (num_experts, bs*seq_len, expert_dim), (num_experts, bs*seq_len, expert_dim)

        gated = up * self.act_fn(gate)

        next_states = torch.bmm(gated, self.down_proj) # (num_experts, bs*seq_len, hidden_size)
        
        next_states = next_states.view(-1, self.hidden_size) # (num_experts*bs*seq_len, hidden_size)

        return next_states
    
class Llama4TextMLP(nn.Module): # Shared expert
    def __init__(self, config):
        super(Llama4TextMLP, self).__init__()

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)

class Llama4TextMoe(nn.Module):

    def __init__(self, config):
        super(Llama4TextMoe, self).__init__()

        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        # define experts
        self.experts = Llama4TextExperts(config)
        # router
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.shared_expert = Llama4TextMLP(config)


    def forward(self, hidden_states):
        
        bs, seq_len, embed_dim = hidden_states.shape

        # (b, s, dim) -> (b*s, dim)
        hidden_states = hidden_states.reshape(-1, embed_dim)

        # Assign each token to a set of experts using a router layer
        router_logits = self.router(hidden_states) # (b*s, num_experts)
        
        #### Step1: Pretend each expert will get all tokens
        tokens_per_expert = bs * seq_len
        #### Step2: Get the top-k for each token
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        #### Step3: Create a matrix of -Infinities, #(Num_experts, bs * seq_len=NUM_TOKENS)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(dim=1, index=router_indices, src=router_top_value).transpose(0,1) #(Num_experts, bs * seq_len=NUM_TOKENS)

        #### Step4: Because we are passing in `ALL TOKENS` to every expert, lets update our router indices to be
        #### indexes from 0 to NUM_TOKENS= bs * seq_len, repeated for EVERY EXPERT! It will look something like this

        ### [0, 1, 2, ... , NUM_TOKENS]
        ### [0, 1, 2, ... , NUM_TOKENS]
        ### [0, 1, 2, ... , NUM_TOKENS]
        ### .
        ### .
        ### .
        ### `NUM_EXPERT TIMES` repeated

        router_indices = torch.arange(tokens_per_expert, device=hidden_states.device).unsqueeze(0).expand(self.num_experts, -1)

        #### Step5: Grab embeddings with these indices, but we have an embed dim as well
        router_indices = router_indices.reshape(-1,1).expand(-1, self.hidden_dim) # (bs * seq_len * num_experts, hidden_dim) where bs * seq_len = num_tokens
        
        #### Step6: Scale our router weights
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype) # (Num_experts, bs * seq_len=NUM_TOKENS)
        
        #### Step 7: gather our hidden states by our router indices, router_in: (b*s*num_experts, hidden_dim)
        router_in = torch.gather(
            input=hidden_states, # (b*s, hidden_dim)
            dim=0,
            index=router_indices # (b*s*num_experts, hidden_dim)
        )
        router_out = router_in * router_scores.reshape(-1,1) # (b*s*num_experts, hidden_dim)

        ### This MOE output
        moe_out = self.experts(router_out)

        ### Shared expert
        shared_expert_out = self.shared_expert(hidden_states)
        
        shared_expert_out.scatter_add_(dim=0, index=router_indices, src=moe_out)
        
        return shared_expert_out
        
class Llama4TextRotaryEmbedding(nn.Module):

    def __init__(self, config, device=None):
        super(Llama4TextRotaryEmbedding, self).__init__()

        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        inv_freq = self._compute_default_rope_parameters(self.config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_default_rope_parameters(self, config, device):

        base = config.rope_theta
        head_dim = config.head_dim
        # initial theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).to(device=device, dtype=torch.float)/ head_dim))
        
        return inv_freq
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        # (b, head_dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, -1).to(x.device)
        
        # position_ids: (b, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].to(torch.float)

        with torch.autocast(device_type=x.device.type, enabled=False):
            freq = (inv_freq_expanded @ position_ids_expanded) # (b, head_dim/2, seq_len)

            freq = freq.transpose(1,2) # (b, seq_len, head_dim/2)
            
            freq_cis = torch.polar(abs=torch.ones_like(freq), angle=freq)

        return freq_cis
    
def apply_rotary_emb(xq, xk, freq_cis):
    # input: xq, xk of shape: (bs, seq_len, num_heads, head_dim)

    xq = xq.float().reshape(*xq.shape[:-1], -1, 2) # (bs, num_heads, seq_len, head_dim/2, 2)
    xk = xk.float().reshape(*xq.shape[:-1], -1, 2) # (bs, num_heads, seq_len, head_dim/2, 2)

    # convert xq, xk to complex number
    xq_complex = torch.view_as_complex(xq) # (bs, num_heads, seq_len, head_dim/2)
    xk_complex = torch.view_as_complex(xq) # (bs, num_heads, seq_len, head_dim/2)

    # freq_cis: # (b, seq_len, head_dim/2), we need an extra dim for num_heads in freq_cis
    xq_rotated = xq_complex * freq_cis[:, :, None, :]
    xk_rotated = xk_complex * freq_cis[:, :, None, :]

    # convert complex numbers back to real numbers
    xq_out = torch.view_as_real(xq_rotated) # (bs, num_heads, seq_len, head_dim/2, 2)
    xk_out = torch.view_as_real(xk_rotated) # (bs, num_heads, seq_len, head_dim/2, 2)

    xq_out = xq_out.flatten(3) # (bs, num_heads, seq_len, head_dim)
    xk_out = xk_out.flatten(3) # (bs, num_heads, seq_len, head_dim)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Llama4TextL2Norm(nn.Module):

    def __init__(self, eps=1e-6):
        super(Llama4TextL2Norm, self).__init__()
        self.eps = eps

    def _norm(self, x):
        ### We want to do x / sqrt((x**2).mean()) along the last dimension
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x)
    
class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(Llama4TextRMSNorm, self).__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
class Cache:
    def __init__(self, config):

        self.seen_tokens = 0
        self.key_cache = [torch.tensor([]) for _ in range(config.num_layers)]
        self.value_cache = [torch.tensor([]) for _ in range(config.num_layers)]

    def __repr__(self):
        return f"Dynamic Cache(Num_Layers: {len(self.key_cache)} | Cached Tokens: {self.key_cache[0].shape[2]})"
    
    def update(self, key_states, value_states, layer_idx):
        ### (bs, num_heads, seq_len, head_dim)

        if layer_idx == 0:
            self.seen_tokens += 1

        self.key_cache[layer_idx] = torch.cat(self.key_cache[layer_idx], key_states, dim=-2) # concat along seq_len dim
        self.value_cache[layer_idx] = torch.cat(self.value_cache[layer_idx], value_states, dim=-2) # concat along seq_len dim

        return self.key_cache, self.value_cache

    
    def get_seq_len(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[-2] if self.key_cache[layer_idx].numel() !=0 else 0
    

class Llama4TextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super(Llama4TextAttention, self).__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = int((layer_idx+1) % 4 == 0)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm()
        else:
            self.qk_norm = None

    def _repeat_kv(self, hidden_states, n_rep):
        batch, heads, seq_len, embed_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, heads, n_rep, seq_len, embed_dim)
        hidden_states = hidden_states.reshape(batch, heads * n_rep, seq_len, embed_dim)
        return hidden_states

    def forward(self,
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                past_key_value=None,
                cache_position=None):
        
        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.head_dim) # adding dummy dim for number of heads

        query_states = self.q_proj(hidden_states).reshape(hidden_shape)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose(1,2 ) # (bs, num_kv_heads, seq_len, head_dim)

        if self.use_rope:
            query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings.to(query_states.device))

        if self.qk_norm is not None:
            query_states = self.qk_norm(query_states)
            key_states = query_states(key_states)

        cache_position = cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1) # this is just token position in sequence
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((cache_position.float() + 1) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1,2) # (bs, num_attn_heads, seq_len, head_dim)
        key_states = key_states.transpose(1,2) # (bs, num_kv_heads, seq_len, head_dim)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        ### Group Query Attention (GQA)
        ### Step1: Repeat Keys/Values Heads to match the Query Heads
        ### Query: (b, attn_head, seq_len, head_dim)
        ### Key: (b, k_v_heads, seq_len, head_dim)
        ### value: (b, k_v_heads, seq_len, head_dim)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups) # (b, attn_head, seq_len, head_dim)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups) # (b, attn_head, seq_len, head_dim)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) # (b, attn_head, seq_len, seq_len)
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:

            causal_mask = attention_mask[:, :, :, :key_states.shape[0]] # ?? Sachin: what's this dimension

            attn_weights = attn_weights + causal_mask
        
        attn_weights = attn_weights.float().softmax(dim=-1).to(query_states.dtype) # (b, attn_head, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, value_states) # (b, attn_head, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).flatten(2) # (b, seq_len, num_attn_heads*head_dim)

        attn_output = self.o_proj(attn_output) # (b, seq_len, hidden_dim)

        return attn_output



if __name__ == "__main__":

    #config = Llama4TextConfig(hidden_size=768,
    #                          intermediate_size=768 * 2,
    #                          intermediate_size_mlp=768 * 2,
    #                          num_experts_per_tok= 2,
    #                          head_dim=64
    #                          )

    config = Llama4TextConfig()

    rope = Llama4TextRotaryEmbedding(config)
    bs, seq_len, emb_dim = 2, 16, 5120
    #num_heads = 12
    #assert emb_dim % num_heads == 0, "number of heads must divide emd dimension"
    #head_dim = emb_dim // num_heads
    x = torch.randn(bs, seq_len, emb_dim)
    position_ids = torch.arange(0,seq_len, dtype=torch.int).repeat(bs,1)
    freq_cis = rope(x, position_ids)
    attn = Llama4TextAttention(config, layer_idx=0)
    attn(x, position_embeddings=freq_cis, cache_position=position_ids[0])

    #xq, xk = apply_rotary_emb(x, x, freq_cis)
    #print(xq.shape, xk.shape)

    #x = torch.randn(bs, seq_len, emb_dim)
    #position_ids = torch.arange(0,seq_len, dtype=torch.int).repeat(bs,1)
    #moe = Llama4TextMoe(config=config)
    #moe(x)

