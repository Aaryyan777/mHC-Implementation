"""
Implementation of Manifold-Constrained Hyper-Connections (mHC) from the research paper.
This implementation includes the core components of mHC: doubly stochastic matrix constraints,
Sinkhorn-Knopp algorithm, and the complete mHC layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
            matrix: Input matrix of shape (batch_size, n, n) or (n, n)
            
        Returns:
            Doubly stochastic matrix of the same shape
        """
        # Ensure all elements are positive using exponent
        matrix = torch.exp(matrix)
        
        # Normalize rows and columns iteratively
        for _ in range(self.iterations):
            # Normalize rows (sum to 1)
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-12)
            # Normalize columns (sum to 1)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-12)
        
        return matrix


class MHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connection layer implementation.
    This replaces the standard residual connection with mHC.
    """
    def __init__(self, input_dim, expansion_rate=4, hidden_dim=None):
        super(MHCLayer, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate  # Stream expansion factor
        self.hidden_dim = hidden_dim or input_dim
        
        # Define dimensions
        self.stream_dim = expansion_rate * input_dim  # Expanded stream dimension
        
        # Linear projections for dynamic mappings
        self.phi_pre = nn.Linear(self.stream_dim, expansion_rate, bias=False)
        self.phi_post = nn.Linear(self.stream_dim, expansion_rate, bias=False)
        self.phi_res = nn.Linear(self.stream_dim, expansion_rate * expansion_rate, bias=False)
        
        # Bias terms for static mappings
        self.bias_pre = nn.Parameter(torch.zeros(1, expansion_rate))
        self.bias_post = nn.Parameter(torch.zeros(1, expansion_rate))
        self.bias_res = nn.Parameter(torch.zeros(1, expansion_rate, expansion_rate))
        
        # Learnable gating factors
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # Sinkhorn-Knopp operator for doubly stochastic constraint
        self.sinkhorn_knopp = SinkhornKnopp(iterations=20)
        
        # RMSNorm for normalization
        self.rms_norm = RMSNorm(self.stream_dim)

    def forward(self, x, layer_func):
        """
        Forward pass of the mHC layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            layer_func: The layer function F (e.g., attention or FFN)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand input to n-stream residual: (batch_size, seq_len, n, input_dim)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        x_expanded = x_expanded.reshape(batch_size, seq_len, self.stream_dim)  # (B, L, n*C)
        
        # Apply RMSNorm
        x_norm = self.rms_norm(x_expanded)  # (B, L, n*C)
        
        # Compute dynamic mappings using linear projections
        dynamic_pre = self.phi_pre(x_norm)  # (B, L, n)
        dynamic_post = self.phi_post(x_norm)  # (B, L, n)
        dynamic_res = self.phi_res(x_norm)  # (B, L, n*n)
        dynamic_res = dynamic_res.view(batch_size, seq_len, self.n, self.n)  # (B, L, n, n)
        
        # Compute full mappings with gating
        H_pre = self.alpha_pre * dynamic_pre + self.bias_pre  # (B, L, n)
        H_post = self.alpha_post * dynamic_post + self.bias_post  # (B, L, n)
        H_res = self.alpha_res * dynamic_res + self.bias_res  # (B, L, n, n)
        
        # Apply constraints: sigmoid for H_pre and H_post, Sinkhorn-Knopp for H_res
        H_pre = torch.sigmoid(H_pre)  # (B, L, n) - ensures non-negativity
        H_post = 2 * torch.sigmoid(H_post)  # (B, L, n) - ensures non-negativity
        H_res = self.sinkhorn_knopp(H_res)  # (B, L, n, n) - doubly stochastic
        
        # Reshape x for processing
        x_streams = x_expanded.view(batch_size, seq_len, self.n, self.input_dim)  # (B, L, n, C)
        
        # Compute H_pre @ x (aggregate features from n-stream to C-dim layer input)
        H_pre_expanded = H_pre.unsqueeze(-1)  # (B, L, n, 1)
        x_pre = x_streams * H_pre_expanded  # (B, L, n, C)
        x_pre = x_pre.sum(dim=2)  # (B, L, C) - sum over streams
        
        # Apply the layer function F
        F_output = layer_func(x_pre)  # (B, L, C)
        
        # Compute H_post^T @ F_output and H_res @ x
        F_output_expanded = F_output.unsqueeze(2).expand(-1, -1, self.n, -1)  # (B, L, n, C)
        H_post_expanded = H_post.unsqueeze(-1)  # (B, L, n, 1)
        H_post_F = F_output_expanded * H_post_expanded  # (B, L, n, C)
        
        # Compute H_res @ x (mix features within residual stream)
        x_res = torch.matmul(H_res, x_streams)  # (B, L, n, C)
        
        # Combine: H_res @ x + H_post^T @ F_output
        combined = x_res + H_post_F  # (B, L, n, C)
        
        # Average over streams to get output
        output = combined.mean(dim=2)  # (B, L, C)
        
        return output


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


class MHCTransformerBlock(nn.Module):
    """
    A complete Transformer block using mHC for both attention and FFN layers
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, expansion_rate=4):
        super(MHCTransformerBlock, self).__init__()
        
        # Multi-head attention with mHC
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mhc_attn = MHCLayer(d_model, expansion_rate)
        
        # Feed-forward network with mHC
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mhc_ffn = MHCLayer(d_model, expansion_rate)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention with mHC
        def attention_func(x):
            attn_output, _ = self.self_attn(x, x, x)
            return self.dropout1(attn_output)
        
        attn_output = self.mhc_attn(src, attention_func)
        src = src + attn_output
        src = self.norm1(src)

        # Feed-forward with mHC
        def ffn_func(x):
            ffn_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
            return self.dropout2(ffn_output)
        
        ffn_output = self.mhc_ffn(src, ffn_func)
        src = src + ffn_output
        src = self.norm2(src)
        
        return src


class MHCModel(nn.Module):
    """
    Complete model using mHC layers
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, expansion_rate=4):
        super(MHCModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            MHCTransformerBlock(d_model, nhead, dim_feedforward, dropout, expansion_rate)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        output = self.fc_out(x)
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def create_mhc_model(vocab_size, d_model=512, nhead=8, num_layers=6, 
                     dim_feedforward=2048, dropout=0.1, expansion_rate=4):
    """
    Helper function to create an mHC model
    """
    return MHCModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        expansion_rate=expansion_rate
    )


# Example usage and testing
if __name__ == "__main__":
    # Create a simple model for testing
    model = create_mhc_model(vocab_size=1000, d_model=256, nhead=4, num_layers=2, expansion_rate=4)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test Sinkhorn-Knopp
    sinkhorn = SinkhornKnopp(iterations=20)
    test_matrix = torch.randn(4, 4)
    doubly_stochastic = sinkhorn(test_matrix)
    
    print(f"Original matrix row sums: {test_matrix.sum(dim=1)}")
    print(f"Original matrix col sums: {test_matrix.sum(dim=0)}")
    print(f"Doubly stochastic row sums: {doubly_stochastic.sum(dim=1)}")
    print(f"Doubly stochastic col sums: {doubly_stochastic.sum(dim=0)}")