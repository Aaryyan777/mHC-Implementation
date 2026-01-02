"""
Detailed implementation of mHC parameterization and mapping functions
following the exact equations from the research paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHCRaw(nn.Module):
    """
    Raw mHC implementation following the exact equations from the paper
    """
    def __init__(self, input_dim, expansion_rate=4):
        super(MHCRaw, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate  # Stream expansion factor
        self.stream_dim = expansion_rate * input_dim  # Expanded stream dimension
        
        # Linear projections for dynamic mappings (phi parameters from Eq. 7)
        self.phi_pre = nn.Parameter(torch.randn(input_dim * expansion_rate, expansion_rate) * 0.02)
        self.phi_post = nn.Parameter(torch.randn(input_dim * expansion_rate, expansion_rate) * 0.02)
        self.phi_res = nn.Parameter(torch.randn(input_dim * expansion_rate, expansion_rate * expansion_rate) * 0.02)
        
        # Bias terms for static mappings (b parameters from Eq. 7)
        self.bias_pre = nn.Parameter(torch.zeros(1, expansion_rate))
        self.bias_post = nn.Parameter(torch.zeros(1, expansion_rate))
        self.bias_res = nn.Parameter(torch.zeros(1, expansion_rate, expansion_rate))
        
        # Learnable gating factors (alpha parameters from Eq. 5 & 7)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # Additional parameters for tanh activation (theta parameters from Eq. 5)
        # These are now replaced with the linear projections in Eq. 7
        self.theta_pre = nn.Parameter(torch.randn(1, input_dim) * 0.02)
        self.theta_post = nn.Parameter(torch.randn(1, input_dim) * 0.02)
        self.theta_res = nn.Parameter(torch.randn(expansion_rate, input_dim) * 0.02)
        
        # Bias terms for tanh activation (b parameters from Eq. 5)
        self.b_pre = nn.Parameter(torch.zeros(1, expansion_rate))
        self.b_post = nn.Parameter(torch.zeros(1, expansion_rate))
        self.b_res = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
        
        # Sinkhorn-Knopp operator for doubly stochastic constraint
        self.sinkhorn_knopp = SinkhornKnopp(iterations=20)
        
        # RMSNorm for normalization
        self.rms_norm = RMSNorm(input_dim * expansion_rate)

    def compute_mappings_paper_style(self, x):
        """
        Compute Hpre, Hpost, Hres following the exact equations from the paper (Eq. 5, 7, 8)
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten x to vector form as in Eq. 7
        x_flat = x.view(batch_size, seq_len, -1)  # (B, L, n*C)
        
        # Apply RMSNorm as in Eq. 7
        x_norm = self.rms_norm(x_flat)  # (B, L, n*C)
        
        # Compute dynamic mappings using linear projections (Eq. 7)
        # Using matrix multiplication as in the paper
        dynamic_pre = torch.matmul(x_norm, self.phi_pre)  # (B, L, n)
        dynamic_post = torch.matmul(x_norm, self.phi_post)  # (B, L, n)
        dynamic_res_reshaped = torch.matmul(x_norm, self.phi_res)  # (B, L, n*n)
        dynamic_res = dynamic_res_reshaped.view(batch_size, seq_len, self.n, self.n)  # (B, L, n, n)
        
        # Add bias terms (Eq. 7)
        H_pre_tilde = self.alpha_pre * dynamic_pre + self.bias_pre  # (B, L, n)
        H_post_tilde = self.alpha_post * dynamic_post + self.bias_post  # (B, L, n)
        H_res_tilde = self.alpha_res * dynamic_res + self.bias_res  # (B, L, n, n)
        
        # Apply constraints as in Eq. 8
        H_pre = torch.sigmoid(H_pre_tilde)  # (B, L, n) - ensures non-negativity
        H_post = 2 * torch.sigmoid(H_post_tilde)  # (B, L, n) - ensures non-negativity
        H_res = self.sinkhorn_knopp(H_res_tilde)  # (B, L, n, n) - doubly stochastic
        
        return H_pre, H_post, H_res

    def forward(self, x, layer_func):
        """
        Forward pass following Eq. 3 from the paper:
        x_{l+1} = Hres_l * x_l + Hpost^T_l * F(Hpre_l * x_l, W_l)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Expand input to n-stream residual as described in Section 3
        # x_l is expanded from (B, L, C) to (B, L, n, C)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        x_expanded = x_expanded.reshape(batch_size, seq_len, self.n * input_dim)  # (B, L, n*C)
        
        # Compute the three mappings using paper-style computation
        H_pre, H_post, H_res = self.compute_mappings_paper_style(x_expanded)
        
        # Reshape x back for processing
        x_streams = x_expanded.view(batch_size, seq_len, self.n, input_dim)  # (B, L, n, C)
        
        # Compute H_pre @ x (aggregate features from n-stream to C-dim layer input)
        H_pre_expanded = H_pre.unsqueeze(-1).unsqueeze(-1)  # (B, L, n, 1, 1)
        x_for_layer = (x_streams.unsqueeze(-2) * H_pre_expanded).sum(dim=2)  # (B, L, 1, C) -> (B, L, C)
        
        # Apply the layer function F
        F_output = layer_func(x_for_layer)  # (B, L, C)
        
        # Compute H_post^T @ F_output
        F_expanded = F_output.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        H_post_expanded = H_post.unsqueeze(-1)  # (B, L, n, 1)
        H_post_F = F_expanded * H_post_expanded  # (B, L, n, C)
        
        # Compute H_res @ x (mix features within residual stream)
        x_res = torch.matmul(H_res, x_streams.transpose(-2, -1)).transpose(-2, -1)  # (B, L, n, C)
        
        # Combine: H_res @ x + H_post^T @ F_output
        combined = x_res + H_post_F  # (B, L, n, C)
        
        # Average over streams to get output
        output = combined.mean(dim=2)  # (B, L, C)
        
        return output


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm for projecting a matrix onto the Birkhoff polytope
    (doubly stochastic matrices). This ensures that both row and column sums equal 1.
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
        # Ensure all elements are positive using exponent (as mentioned in paper)
        matrix = torch.exp(matrix)
        
        # Normalize rows and columns iteratively (Eq. 9 in paper)
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


def test_mhc_layer():
    """
    Test the mHC layer implementation
    """
    # Create a simple mHC layer
    mhc_layer = MHCRaw(input_dim=128, expansion_rate=4)
    
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
    
    # Test Sinkhorn-Knopp constraints
    test_matrix = torch.randn(4, 4)
    sinkhorn = SinkhornKnopp(iterations=20)
    doubly_stochastic = sinkhorn(test_matrix)
    
    print(f"\nSinkhorn-Knopp validation:")
    print(f"Row sums: {doubly_stochastic.sum(dim=1)}")
    print(f"Col sums: {doubly_stochastic.sum(dim=0)}")
    print(f"All values positive: {(doubly_stochastic > 0).all().item()}")


if __name__ == "__main__":
    test_mhc_layer()