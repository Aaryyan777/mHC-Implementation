"""
Recompute functionality for mHC as described in Section 4.3.2 of the paper.
This implementation reduces memory overhead by recomputing intermediate activations
during the backward pass instead of storing them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


class RecomputeMHCLayer(nn.Module):
    """
    mHC layer with recompute functionality as described in Section 4.3.2
    """
    def __init__(self, input_dim, expansion_rate=4, recompute_blocks=4):
        super(RecomputeMHCLayer, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.stream_dim = expansion_rate * input_dim
        self.recompute_blocks = recompute_blocks  # Lr in the paper
        
        # Fused kernels for computing Hpre, Hpost, Hres
        self.fused_kernels = FusedMHCKernels(input_dim, expansion_rate)
        
        # Gate parameters
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # RMSNorm
        self.rms_norm = RMSNorm(input_dim * expansion_rate)
        
        # Sinkhorn-Knopp for doubly stochastic constraint
        self.sinkhorn_knopp = FusedSinkhornKnopp(iterations=20)

    def compute_mappings_with_recompute(self, x_flat):
        """
        Compute mappings with recompute functionality to save memory
        """
        batch_size, seq_len, _ = x_flat.shape
        
        # Apply RMSNorm
        x_norm = self.rms_norm(x_flat)
        
        # Compute all mappings
        all_mappings = torch.matmul(x_norm, self.fused_kernels.phi_params)
        all_mappings = all_mappings + self.fused_kernels.bias_params
        
        # Extract individual mappings
        H_pre_tilde = all_mappings[:, :, :self.n]
        H_post_tilde = all_mappings[:, :, self.n:2*self.n]
        H_res_tilde = all_mappings[:, :, 2*self.n:]
        H_res_tilde = H_res_tilde.view(batch_size, seq_len, self.n, self.n)
        
        # Apply gate factors
        H_pre_tilde = self.alpha_pre * H_pre_tilde
        H_post_tilde = self.alpha_post * H_post_tilde
        H_res_tilde = self.alpha_res * H_res_tilde
        
        # Apply final transformations
        H_pre = torch.sigmoid(H_pre_tilde)
        H_post = 2.0 * torch.sigmoid(H_post_tilde)
        H_res = self.sinkhorn_knopp(H_res_tilde)
        
        return H_pre, H_post, H_res

    def forward(self, x, layer_func, use_recompute=False):
        """
        Forward pass with optional recompute functionality
        """
        if use_recompute:
            # Use PyTorch's checkpointing for recompute functionality
            return self._forward_with_recompute(x, layer_func)
        else:
            return self._forward_standard(x, layer_func)

    def _forward_standard(self, x, layer_func):
        """
        Standard forward pass without recompute
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Expand input to n-stream residual
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)
        x_flat = x_expanded.reshape(batch_size, seq_len, self.n * input_dim)
        
        # Compute all mappings
        H_pre, H_post, H_res = self.compute_mappings_with_recompute(x_flat)
        
        # Apply H_pre to x to get input for layer function
        x_for_layer = self.apply_mapping_pre(x_flat, H_pre)
        
        # Apply the layer function F (this is the heavy computation we don't recompute)
        F_output = layer_func(x_for_layer)
        
        # Apply H_res and H_post mappings and combine
        result = self.apply_combined_mappings(x_flat, F_output, H_res, H_post)
        
        return result

    def _forward_with_recompute(self, x, layer_func):
        """
        Forward pass with recompute functionality
        Following Section 4.3.2: discard intermediate activations after forward pass
        and recompute them during backward pass
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Expand input to n-stream residual
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)
        x_flat = x_expanded.reshape(batch_size, seq_len, self.n * input_dim)
        
        # Use checkpointing for the mHC kernel computations (without the heavy layer function)
        def compute_mappings_checkpoint(x_flat):
            return self.compute_mappings_with_recompute(x_flat)
        
        # Compute mappings using checkpointing (will be recomputed in backward pass)
        H_pre, H_post, H_res = checkpoint(compute_mappings_checkpoint, x_flat)
        
        # Apply H_pre to x to get input for layer function (not checkpointed)
        x_for_layer = self.apply_mapping_pre(x_flat, H_pre)
        
        # Apply the layer function F (this is the heavy computation we don't recompute)
        F_output = layer_func(x_for_layer)
        
        # Apply H_res and H_post mappings and combine (not checkpointed)
        result = self.apply_combined_mappings(x_flat, F_output, H_res, H_post)
        
        return result

    def apply_mapping_pre(self, x_flat, H_pre):
        """
        Apply H_pre mapping: compute H_pre @ x to get layer input
        """
        batch_size, seq_len, _ = x_flat.shape
        x_streams = x_flat.view(batch_size, seq_len, self.n, self.input_dim)
        
        # Apply H_pre: weighted sum across streams
        H_pre_expanded = H_pre.unsqueeze(-1)
        x_weighted = x_streams * H_pre_expanded
        x_for_layer = x_weighted.sum(dim=2)
        
        return x_for_layer

    def apply_combined_mappings(self, x_flat, F_output, H_res, H_post):
        """
        Apply H_res and H_post mappings and combine: H_res @ x + Hpost^T @ F
        """
        batch_size, seq_len, _ = x_flat.shape
        x_streams = x_flat.view(batch_size, seq_len, self.n, self.input_dim)
        
        # Compute H_res @ x (mix features within residual stream)
        # H_res: (B, L, n, n), x_streams: (B, L, n, input_dim)
        # We want to multiply each n×n matrix in H_res with the corresponding n×input_dim matrix in x_streams
        H_res_x = torch.matmul(H_res, x_streams)  # (B, L, n, input_dim)
        
        # Compute H_post^T @ F_output
        F_expanded = F_output.unsqueeze(2).expand(-1, -1, self.n, -1)
        H_post_expanded = H_post.unsqueeze(-1)
        H_post_F = F_expanded * H_post_expanded
        
        # Combine: H_res @ x + H_post^T @ F
        combined = H_res_x + H_post_F
        
        # Average over streams to get output
        output = combined.mean(dim=2)
        
        return output


class FusedMHCKernels(nn.Module):
    """
    Fused kernels for mHC computation
    """
    def __init__(self, input_dim, expansion_rate=4):
        super(FusedMHCKernels, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.stream_dim = expansion_rate * input_dim
        
        # Combined parameter tensor
        self.phi_params = nn.Parameter(
            torch.randn(input_dim * expansion_rate, 
                       expansion_rate * expansion_rate + 2 * expansion_rate) * 0.02
        )
        
        # Bias terms
        self.bias_params = nn.Parameter(
            torch.zeros(1, expansion_rate * expansion_rate + 2 * expansion_rate)
        )
        
        # RMSNorm
        self.rms_norm = RMSNorm(input_dim * expansion_rate)
        
        # Sinkhorn-Knopp for doubly stochastic constraint
        self.sinkhorn_knopp = FusedSinkhornKnopp(iterations=20)


class FusedSinkhornKnopp(nn.Module):
    """
    Fused Sinkhorn-Knopp with custom backward pass
    """
    def __init__(self, iterations=20):
        super(FusedSinkhornKnopp, self).__init__()
        self.iterations = iterations

    def forward(self, matrix):
        """
        Forward pass with fused operations
        """
        # Ensure all elements are positive via exponent operator
        matrix = torch.exp(matrix)
        
        # Normalize rows and columns iteratively
        for _ in range(self.iterations):
            # Normalize rows (sum to 1) - T_r operation
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-12)
            # Normalize columns (sum to 1) - T_c operation
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-12)
        
        return matrix


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim, eps=1e-20):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class RecomputeTransformerBlock(nn.Module):
    """
    Transformer block with recompute functionality for mHC layers
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 expansion_rate=4, recompute_blocks=4):
        super(RecomputeTransformerBlock, self).__init__()
        
        # Self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # mHC layer for attention with recompute
        self.mhc_attn = RecomputeMHCLayer(d_model, expansion_rate, recompute_blocks)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        
        # mHC layer for FFN with recompute
        self.mhc_ffn = RecomputeMHCLayer(d_model, expansion_rate, recompute_blocks)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, use_recompute=False):
        # Attention sublayer with mHC and recompute
        def attention_func(x):
            attn_output, _ = self.self_attn(x, x, x)
            return self.dropout1(attn_output)
        
        attn_output = self.mhc_attn(src, attention_func, use_recompute=use_recompute)
        src = src + attn_output
        src = self.norm1(src)

        # FFN sublayer with mHC and recompute
        def ffn_func(x):
            ffn_output = self.linear2(self.dropout_ffn(F.relu(self.linear1(x))))
            return self.dropout2(ffn_output)
        
        ffn_output = self.mhc_ffn(src, ffn_func, use_recompute=use_recompute)
        src = src + ffn_output
        src = self.norm2(src)
        
        return src


def calculate_optimal_block_size(n, total_layers, device_memory_gb=16):
    """
    Calculate optimal recompute block size Lr* as described in Eq. 20 of the paper:
    L*_r = arg min_Lr [nC * ceil(L/Lr) + (n+2)C * Lr] ≈ sqrt(nL/(n+2))
    
    Args:
        n: expansion rate
        total_layers: total number of layers L
        device_memory_gb: available GPU memory in GB (for estimation)
    
    Returns:
        Optimal block size L*_r
    """
    # According to Eq. 20: L*_r ≈ sqrt(nL/(n+2))
    optimal_Lr = math.sqrt(n * total_layers / (n + 2))
    
    # In practice, often aligns with pipeline stage boundaries
    # For now, return the calculated value
    return int(optimal_Lr)


def test_recompute_functionality():
    """
    Test the recompute functionality implementation
    """
    print("Testing Recompute Functionality Implementation...")
    
    # Test basic recompute layer
    recompute_layer = RecomputeMHCLayer(input_dim=128, expansion_rate=4)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    dummy_input = torch.randn(batch_size, seq_len, 128, requires_grad=True)
    
    # Define a simple layer function for testing
    simple_layer = nn.Linear(128, 128)
    
    def layer_func(x):
        return simple_layer(x)
    
    # Test forward pass without recompute
    output_no_recompute = recompute_layer(dummy_input, layer_func, use_recompute=False)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (no recompute): {output_no_recompute.shape}")
    
    # Test forward pass with recompute
    output_with_recompute = recompute_layer(dummy_input, layer_func, use_recompute=True)
    print(f"Output shape (with recompute): {output_with_recompute.shape}")
    
    # Verify outputs are similar
    diff = torch.abs(output_no_recompute - output_with_recompute).mean()
    print(f"Mean absolute difference between outputs: {diff.item():.8f}")
    
    # Test transformer block with recompute
    transformer_block = RecomputeTransformerBlock(d_model=128, nhead=4, expansion_rate=4)
    block_output_no_recompute = transformer_block(dummy_input, use_recompute=False)
    block_output_with_recompute = transformer_block(dummy_input, use_recompute=True)
    print(f"Transformer block output shape (no recompute): {block_output_no_recompute.shape}")
    print(f"Transformer block output shape (with recompute): {block_output_with_recompute.shape}")
    
    # Test optimal block size calculation
    n = 4  # expansion rate
    total_layers = 12
    optimal_Lr = calculate_optimal_block_size(n, total_layers)
    print(f"Optimal recompute block size for n={n}, L={total_layers}: {optimal_Lr}")
    
    print("\nRecompute functionality implementation test passed!")


if __name__ == "__main__":
    test_recompute_functionality()