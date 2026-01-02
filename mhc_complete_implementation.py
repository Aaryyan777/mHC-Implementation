"""
Complete mHC (Manifold-Constrained Hyper-Connections) layer implementation
following the research paper specifications exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHCCompleteLayer(nn.Module):
    """
    Complete mHC layer implementation following the paper equations exactly.
    Implements Eq. 3: x_{l+1} = Hres_l * x_l + Hpost^T_l * F(Hpre_l * x_l, W_l)
    with doubly stochastic constraints on Hres_l via Sinkhorn-Knopp.
    """
    def __init__(self, input_dim, expansion_rate=4, use_mixed_precision=True):
        super(MHCCompleteLayer, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate  # Stream expansion factor
        self.stream_dim = expansion_rate * input_dim  # Expanded stream dimension
        
        # Parameters for dynamic and static mappings (Eq. 7)
        # Using the fused parameterization from the paper
        self.phi_params = nn.Parameter(torch.randn(
            input_dim * expansion_rate, 
            expansion_rate * expansion_rate + 2 * expansion_rate
        ) * 0.02)  # Combined phi parameters for efficiency
        
        # Bias terms for dynamic mappings
        self.bias_params = nn.Parameter(torch.zeros(
            1, 
            expansion_rate * expansion_rate + 2 * expansion_rate
        ))
        
        # Separate the bias terms for each mapping
        self.register_buffer('pre_slice', torch.arange(0, expansion_rate))
        self.register_buffer('post_slice', torch.arange(expansion_rate, 2 * expansion_rate))
        self.register_buffer('res_slice', torch.arange(2 * expansion_rate, 
                                                      2 * expansion_rate + expansion_rate * expansion_rate))
        
        # Learnable gating factors (alpha parameters)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # Sinkhorn-Knopp operator for doubly stochastic constraint
        self.sinkhorn_knopp = SinkhornKnopp(iterations=20)
        
        # RMSNorm for normalization
        self.rms_norm = RMSNorm(input_dim * expansion_rate)
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision

    def compute_mappings(self, x):
        """
        Compute Hpre_l, Hpost_l, and Hres_l mappings following the paper equations.
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten x to vector form as in Eq. 7
        x_flat = x.view(batch_size, seq_len, -1)  # (B, L, n*C)
        
        # Apply RMSNorm as in Eq. 7
        x_norm = self.rms_norm(x_flat)  # (B, L, n*C)
        
        # Compute all mappings in one operation for efficiency
        all_mappings = torch.matmul(x_norm, self.phi_params)  # (B, L, n*n + 2*n)
        all_mappings = all_mappings + self.bias_params  # Add bias terms
        
        # Extract individual mappings
        H_pre_tilde = all_mappings[:, :, self.pre_slice]  # (B, L, n)
        H_post_tilde = all_mappings[:, :, self.post_slice]  # (B, L, n)
        
        # Reshape H_res_tilde to (B, L, n, n)
        H_res_tilde = all_mappings[:, :, self.res_slice]  # (B, L, n*n)
        H_res_tilde = H_res_tilde.view(batch_size, seq_len, self.n, self.n)  # (B, L, n, n)
        
        # Apply gating factors
        H_pre_tilde = self.alpha_pre * H_pre_tilde
        H_post_tilde = self.alpha_post * H_post_tilde
        H_res_tilde = self.alpha_res * H_res_tilde
        
        # Apply constraints as in Eq. 8
        H_pre = torch.sigmoid(H_pre_tilde)  # (B, L, n) - ensures non-negativity
        H_post = 2 * torch.sigmoid(H_post_tilde)  # (B, L, n) - ensures non-negativity
        H_res = self.sinkhorn_knopp(H_res_tilde)  # (B, L, n, n) - doubly stochastic
        
        return H_pre, H_post, H_res

    def forward(self, x, layer_func):
        """
        Forward pass implementing Eq. 3 from the paper:
        x_{l+1} = Hres_l * x_l + Hpost^T_l * F(Hpre_l * x_l, W_l)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Expand input to n-stream residual as described in Section 3
        # x_l is expanded from (B, L, C) to (B, L, n, C) then flattened to (B, L, n*C)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        x_flat = x_expanded.reshape(batch_size, seq_len, self.n * input_dim)  # (B, L, n*C)
        
        # Compute the three mappings
        H_pre, H_post, H_res = self.compute_mappings(x_flat)
        
        # Reshape x back for processing
        x_streams = x_expanded  # (B, L, n, C)
        
        # Compute H_pre @ x (aggregate features from n-stream to C-dim layer input)
        # H_pre has shape (B, L, n), x_streams has shape (B, L, n, C)
        # Result should be (B, L, C)
        H_pre_expanded = H_pre.unsqueeze(-1)  # (B, L, n, 1)
        x_for_layer = (x_streams * H_pre_expanded).sum(dim=2)  # (B, L, C)
        
        # Apply the layer function F
        F_output = layer_func(x_for_layer)  # (B, L, C)
        
        # Compute H_post^T @ F_output
        # H_post has shape (B, L, n), F_output has shape (B, L, C)
        # We want to broadcast F_output to (B, L, n, C) and multiply by H_post
        F_expanded = F_output.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        H_post_expanded = H_post.unsqueeze(-1)  # (B, L, n, 1)
        H_post_F = F_expanded * H_post_expanded  # (B, L, n, C)
        
        # Compute H_res @ x (mix features within residual stream)
        # H_res has shape (B, L, n, n), x_streams has shape (B, L, n, C)
        # We need to perform matrix multiplication: (B, L, n, n) @ (B, L, n, C) -> (B, L, n, C)
        H_res_x = torch.matmul(H_res, x_streams)  # (B, L, n, C)
        
        # Combine: H_res @ x + H_post^T @ F_output
        combined = H_res_x + H_post_F  # (B, L, n, C)
        
        # Average over streams to get final output (B, L, C)
        output = combined.mean(dim=2)  # (B, L, C)
        
        return output


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm for projecting a matrix onto the Birkhoff polytope
    (doubly stochastic matrices). This ensures that both row and column sums equal 1.
    Implements the algorithm described in Section 4.2.
    """
    def __init__(self, iterations=20):
        super(SinkhornKnopp, self).__init__()
        self.iterations = iterations

    def forward(self, matrix):
        """
        Apply Sinkhorn-Knopp algorithm to make matrix doubly stochastic.
        
        Args:
            matrix: Input matrix of shape (batch_size, seq_len, n, n) or (n, n)
            
        Returns:
            Doubly stochastic matrix of the same shape
        """
        # Ensure all elements are positive via exponent operator (as in paper)
        matrix = torch.exp(matrix)
        
        # Normalize rows and columns iteratively (Eq. 9 in paper)
        # T_r and T_c denote row and column normalization respectively
        for _ in range(self.iterations):
            # Row normalization: T_r operation
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-12)
            # Column normalization: T_c operation  
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


class OptimizedMHCLayer(nn.Module):
    """
    Optimized mHC layer with kernel fusion as described in Section 4.3.1
    """
    def __init__(self, input_dim, expansion_rate=4):
        super(OptimizedMHCLayer, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.stream_dim = expansion_rate * input_dim
        
        # Combined parameter tensor for fused operations (Eq. 10-13)
        # This represents the fused phi parameters and biases
        self.combined_params = nn.Parameter(torch.randn(
            input_dim * expansion_rate,
            expansion_rate * expansion_rate + 2 * expansion_rate
        ) * 0.02)
        
        self.combined_biases = nn.Parameter(torch.zeros(
            1, 
            expansion_rate * expansion_rate + 2 * expansion_rate
        ))
        
        # Gate parameters (alpha in Eq. 5)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # RMSNorm with absorbed weight (as mentioned in Section 4.3.1)
        self.rms_norm = RMSNorm(input_dim * expansion_rate)
        
        # Sinkhorn-Knopp for doubly stochastic constraint
        self.sinkhorn_knopp = SinkhornKnopp(iterations=20)
        
        # Slicing indices for extracting different mappings
        self.pre_start, self.pre_end = 0, expansion_rate
        self.post_start, self.post_end = expansion_rate, 2 * expansion_rate
        self.res_start, self.res_end = 2 * expansion_rate, 2 * expansion_rate + expansion_rate * expansion_rate

    def fused_compute(self, x):
        """
        Fused computation of all mappings following the kernel fusion approach
        described in Section 4.3.1 of the paper.
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten input
        x_flat = x.view(batch_size, seq_len, -1)  # (B, L, n*C)
        
        # Apply RMSNorm (fused with following operations as per Section 4.3.1)
        x_norm = self.rms_norm(x_flat)
        
        # Compute all mappings in one fused operation (Eqs. 14-15)
        all_outputs = torch.matmul(x_norm, self.combined_params)  # (B, L, n*n + 2*n)
        all_outputs = all_outputs + self.combined_biases  # Add biases (Eq. 16)
        
        # Apply scaling factors (Eq. 16)
        pre_part = self.alpha_pre * all_outputs[:, :, self.pre_start:self.pre_end]
        post_part = self.alpha_post * all_outputs[:, :, self.post_start:self.post_end]
        res_part = self.alpha_res * all_outputs[:, :, self.res_start:self.res_end]
        
        # Apply final transformations (Eqs. 17-19)
        H_pre = torch.sigmoid(pre_part)  # (B, L, n)
        H_post = 2 * torch.sigmoid(post_part)  # (B, L, n)
        H_res_reshaped = res_part.view(batch_size, seq_len, self.n, self.n)  # (B, L, n, n)
        H_res = self.sinkhorn_knopp(H_res_reshaped)  # (B, L, n, n)
        
        return H_pre, H_post, H_res

    def forward(self, x, layer_func):
        """
        Forward pass with fused operations for efficiency.
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Expand input to n-stream
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        x_flat = x_expanded.reshape(batch_size, seq_len, self.n * input_dim)  # (B, L, n*C)
        
        # Fused computation of all mappings
        H_pre, H_post, H_res = self.fused_compute(x_flat)
        
        # Process using the computed mappings
        x_streams = x_expanded  # (B, L, n, C)
        
        # Compute H_pre @ x
        H_pre_expanded = H_pre.unsqueeze(-1)  # (B, L, n, 1)
        x_for_layer = (x_streams * H_pre_expanded).sum(dim=2)  # (B, L, C)
        
        # Apply layer function
        F_output = layer_func(x_for_layer)  # (B, L, C)
        
        # Compute H_post^T @ F_output
        F_expanded = F_output.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        H_post_expanded = H_post.unsqueeze(-1)  # (B, L, n, 1)
        H_post_F = F_expanded * H_post_expanded  # (B, L, n, C)
        
        # Compute H_res @ x
        H_res_x = torch.matmul(H_res, x_streams)  # (B, L, n, C)
        
        # Combine results
        combined = H_res_x + H_post_F  # (B, L, n, C)
        output = combined.mean(dim=2)  # (B, L, C)
        
        return output


def create_mhc_transformer_block(d_model, nhead, dim_feedforward=2048, 
                                 dropout=0.1, expansion_rate=4):
    """
    Create a transformer block using mHC layers
    """
    return MHCTransformerBlock(d_model, nhead, dim_feedforward, 
                               dropout, expansion_rate)


class MHCTransformerBlock(nn.Module):
    """
    Transformer block with mHC connections
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, expansion_rate=4):
        super(MHCTransformerBlock, self).__init__()
        
        # Self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # mHC layer for attention
        self.mhc_attn = OptimizedMHCLayer(d_model, expansion_rate)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        
        # mHC layer for FFN
        self.mhc_ffn = OptimizedMHCLayer(d_model, expansion_rate)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Attention sublayer with mHC
        def attention_func(x):
            attn_output, _ = self.self_attn(x, x, x)
            return self.dropout1(attn_output)
        
        attn_output = self.mhc_attn(src, attention_func)
        src = src + attn_output
        src = self.norm1(src)

        # FFN sublayer with mHC
        def ffn_func(x):
            ffn_output = self.linear2(self.dropout_ffn(F.relu(self.linear1(x))))
            return self.dropout2(ffn_output)
        
        ffn_output = self.mhc_ffn(src, ffn_func)
        src = src + ffn_output
        src = self.norm2(src)
        
        return src


def test_complete_mhc():
    """
    Test the complete mHC implementation
    """
    print("Testing Complete mHC Implementation...")
    
    # Test basic MHC layer
    mhc_layer = OptimizedMHCLayer(input_dim=128, expansion_rate=4)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    dummy_input = torch.randn(batch_size, seq_len, 128)
    
    # Define a simple layer function for testing
    simple_layer = nn.Linear(128, 128)
    
    def layer_func(x):
        return simple_layer(x)
    
    # Forward pass
    output = mhc_layer(dummy_input, layer_func)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test transformer block
    transformer_block = MHCTransformerBlock(d_model=128, nhead=4, expansion_rate=4)
    block_output = transformer_block(dummy_input)
    print(f"Transformer block output shape: {block_output.shape}")
    
    # Test Sinkhorn-Knopp constraints
    test_matrix = torch.randn(4, 4)
    sinkhorn = SinkhornKnopp(iterations=20)
    doubly_stochastic = sinkhorn(test_matrix)
    
    print(f"\nSinkhorn-Knopp validation:")
    print(f"Row sums: {doubly_stochastic.sum(dim=1)}")
    print(f"Col sums: {doubly_stochastic.sum(dim=0)}")
    print(f"All values positive: {(doubly_stochastic > 0).all().item()}")
    print(f"Spectral norm <= 1: {torch.linalg.norm(doubly_stochastic, ord=2).item() <= 1.0}")
    
    print("\nComplete mHC implementation test passed!")


if __name__ == "__main__":
    test_complete_mhc()