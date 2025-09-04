import torch
import torch.nn as nn
import torch.nn.functional as F

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # x: (B, seq_len, input_dim)
        # cat_ids: (B, 1, 1) or (B,) embodiment IDs
        if cat_ids.dim() == 3:
            cat_ids = cat_ids.squeeze(-1).squeeze(-1)  # (B, 1, 1) -> (B,)
        elif cat_ids.dim() == 2:
            cat_ids = cat_ids.squeeze(-1)  # (B, 1) -> (B,)
            
        selected_W = self.W[cat_ids]  # (B, input_dim, hidden_dim)
        selected_b = self.b[cat_ids]  # (B, hidden_dim)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

def build_slices_from_lengths(state_composition):
    slices, off = {}, 0
    for k, L in state_composition.items():
        slices[k] = (off, off + int(L))
        off += int(L)
    return slices, off

class PerModalityStateTokenizer(nn.Module):
    """
    Turns a concatenated state tensor into per-modality tokens.
    Keeps embodiment-aware adapters by reusing CategorySpecificMLP.
    No gating here (yet).
    """
    def __init__(self, state_composition, hidden_size, embed_dim, num_embodiments):
        super().__init__()
        self.modalities = state_composition.keys()
        self.in_dims = state_composition
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.num_embodiments = num_embodiments

        # One adapter per modality → (B,S=1,in_dim) -> (B,S=1,embed_dim)
        self.adapters = nn.ModuleDict({
            m: CategorySpecificMLP(num_embodiments, self.in_dims[m], hidden_size, embed_dim)
            for m in self.modalities
        })
        # Additive token-type embedding per modality (broadcast over batch/seq)
        self.type_embed = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) for m in self.modalities
        })

        # slices will be injected from config
        self._slices = None

    def set_slices_from_keys_lengths(self, state_composition):
        slices, total = build_slices_from_lengths(state_composition)
        self._slices = slices
        return total
    
    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, sdict):
        # dict: modality -> (start, end)
        self._slices = dict(sdict)

    def forward(self, state_btD, embodiment_id_bt1):
        """
        state_btD:          (B, T, D_state)  concatenated state per timestep
        embodiment_id_bt1: (B, T, 1) or (B, 1, 1) int IDs for CategorySpecificMLP
        returns: state_tokens (B, T, M, embed_dim)
        """
        assert self._slices is not None, "PerModalityStateTokenizer.slices not set"
        B, T, D = state_btD.shape

        # Make embodiment ids align with (B*T,1,1)
        if embodiment_id_bt1.dim() == 3 and embodiment_id_bt1.size(1) == 1:
            emb_ids = embodiment_id_bt1.expand(B, T, 1)
        else:
            emb_ids = embodiment_id_bt1
        emb_ids = emb_ids.reshape(B*T, 1, 1)

        tokens = []
        for m in self.modalities:
            s, e = self._slices[m]
            x = state_btD[..., s:e]            # (B, T, in_dim_m)
            x = x.reshape(B*T, 1, e - s)       # CategorySpecificMLP expects (B', S=1, in_dim)
            z = self.adapters[m](x, emb_ids)   # (B*T, 1, embed_dim) embodiment-aware
            z = z + self.type_embed[m]         # token-type embedding
            z = z.reshape(B, T, 1, self.embed_dim)
            tokens.append(z)

        state_tokens = torch.cat(tokens, dim=2)  # (B, T, M, D)
        return state_tokens


# def _first_two_linear_layers(module: nn.Module):
#     layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
#     assert len(layers) >= 2, "Expected at least two Linear layers in CategorySpecificMLP"
#     return layers[0], layers[1]

# @torch.no_grad()
# def migrate_fused_to_tokenized(model, modality_slices: dict):
#     """
#     Copy weights from fused state_encoder to per-modality adapters by slicing columns of the first linear layer.
#     Assumes model.action_head.state_encoder and .state_tokenizer exist.
#     """
#     head = model.action_head
#     fused = head.state_encoder
#     tok  = head.state_tokenizer

#     lin1_fused, lin2_fused = _first_two_linear_layers(fused)

#     for m, (s,e) in modality_slices.items():
#         lin1_m, lin2_m = _first_two_linear_layers(tok.adapters[m])

#         # 1) first layer: copy column slice (D, in_dim_m)
#         lin1_m.weight.copy_(lin1_fused.weight[:, s:e])
#         if lin1_m.bias is not None and lin1_fused.bias is not None:
#             lin1_m.bias.copy_(lin1_fused.bias)

#         # 2) second layer: copy whole (D, D)
#         lin2_m.weight.copy_(lin2_fused.weight)
#         if lin2_m.bias is not None and lin2_fused.bias is not None:
#             lin2_m.bias.copy_(lin2_fused.bias)

#         # leave tok.type_embed[m] small-random (good)
#     return model



def _get_category_specific_layers(module: nn.Module):
    """Extract CategorySpecificLinear layers from CategorySpecificMLP"""
    if hasattr(module, 'layer1') and hasattr(module, 'layer2'):
        return module.layer1, module.layer2
    else:
        # Fallback: look for CategorySpecificLinear layers
        layers = [m for m in module.children() if isinstance(m, CategorySpecificLinear)]
        if len(layers) >= 2:
            return layers[0], layers[1]
        else:
            raise ValueError(f"Expected CategorySpecificMLP with layer1 and layer2, got {type(module)}")

@torch.no_grad()
def migrate_fused_to_tokenized(model, modality_slices: dict):
    """
    Copy weights from fused state_encoder to per-modality adapters by slicing weights from CategorySpecificLinear layers.
    This function is optional - if the migration fails, the model will still work with randomly initialized weights.
    """
    head = model.action_head
    
    # Check if we have both old and new components
    if not hasattr(head, 'state_encoder'):
        print("Info: No state_encoder found in model, using randomly initialized state_tokenizer")
        return model
    if not hasattr(head, 'state_tokenizer'):
        print("Info: No state_tokenizer found in model, migration not needed")
        return model
        
    fused = head.state_encoder
    tok  = head.state_tokenizer

    # If the model was loaded from an existing checkpoint that already has per-modality structure,
    # we might not need migration
    try:
        layer1_fused, layer2_fused = _get_category_specific_layers(fused)
    except Exception as e:
        print(f"Info: Could not extract layers from state_encoder (model might already be migrated): {e}")
        return model

    print(f"Migrating weights for modalities: {list(modality_slices.keys())}")
    
    for m, (s, e) in modality_slices.items():
        if m not in tok.adapters:
            print(f"Warning: Modality '{m}' not found in tokenizer adapters")
            continue
            
        try:
            layer1_m, layer2_m = _get_category_specific_layers(tok.adapters[m])

            # Verify dimensions match before copying
            expected_in_dim = e - s
            if layer1_fused.W.shape[1] < e or layer1_m.W.shape[1] != expected_in_dim:
                print(f"Warning: Dimension mismatch for modality '{m}' (expected {expected_in_dim}, got fused={layer1_fused.W.shape[1]}, target={layer1_m.W.shape[1]})")
                continue

            # 1) first layer: copy column slice from W parameter (num_categories, input_dim, hidden_dim)
            # Slice along input_dim (dimension 1) 
            layer1_m.W.data.copy_(layer1_fused.W.data[:, s:e, :])
            layer1_m.b.data.copy_(layer1_fused.b.data)

            # 2) second layer: copy whole weight matrix 
            layer2_m.W.data.copy_(layer2_fused.W.data)
            layer2_m.b.data.copy_(layer2_fused.b.data)
                
            print(f"✓ Successfully migrated weights for modality '{m}' (dims {s}:{e})")
        except Exception as e:
            print(f"Warning: Could not migrate weights for modality '{m}': {e}")
            print("  -> The model will use randomly initialized weights for this modality")

    print("Migration process completed")
    return model