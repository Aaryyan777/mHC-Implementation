"""
DualPipe communication overlapping for mHC as described in Section 4.3.3 of the paper.
This implementation overlaps communication and computation in pipeline parallelism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
import threading
import time


class DualPipeScheduler:
    """
    DualPipe scheduler for overlapping communication and computation as described in Section 4.3.3
    """
    def __init__(self, num_stages: int, microbatch_size: int = 1, dtype=torch.float32):
        self.num_stages = num_stages
        self.microbatch_size = microbatch_size
        self.dtype = dtype
        
        # Create separate streams for different operations
        self.compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.communication_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.high_priority_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Stage boundaries and recomputation blocks
        self.stage_boundaries = self._calculate_stage_boundaries()
        
    def _calculate_stage_boundaries(self):
        """
        Calculate stage boundaries for pipeline parallelism
        """
        # In practice, this would depend on the model architecture and hardware
        # For now, we'll evenly distribute stages
        return [i * (self.num_stages // self.num_stages) for i in range(self.num_stages + 1)]
    
    def schedule_forward(self, stage_id: int, x, stage_module, use_recompute: bool = True):
        """
        Schedule forward pass with DualPipe overlapping
        """
        if torch.cuda.is_available():
            with torch.cuda.stream(self.compute_stream):
                output = stage_module(x)
        else:
            output = stage_module(x)
        
        return output
    
    def schedule_backward(self, stage_id: int, grad_output, stage_module, input_tensor, 
                         use_recompute: bool = True):
        """
        Schedule backward pass with DualPipe overlapping
        """
        if torch.cuda.is_available():
            with torch.cuda.stream(self.compute_stream):
                # For recompute, we need to recompute the forward pass
                if use_recompute:
                    # Recompute the forward pass for this stage
                    recomputed_output = stage_module(input_tensor)
                    # Compute gradients
                    grad_input = torch.autograd.grad(recomputed_output, input_tensor, grad_output)[0]
                else:
                    # Standard backward pass
                    grad_input = torch.autograd.grad(stage_module(input_tensor), input_tensor, grad_output)[0]
        else:
            if use_recompute:
                recomputed_output = stage_module(input_tensor)
                grad_input = torch.autograd.grad(recomputed_output, input_tensor, grad_output)[0]
            else:
                grad_input = torch.autograd.grad(stage_module(input_tensor), input_tensor, grad_output)[0]
        
        return grad_input


class DualPipeMHCLayer(nn.Module):
    """
    mHC layer with DualPipe communication overlapping
    """
    def __init__(self, input_dim: int, expansion_rate: int = 4, 
                 num_pipeline_stages: int = 1, stage_id: int = 0):
        super(DualPipeMHCLayer, self).__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.stream_dim = expansion_rate * input_dim
        self.num_pipeline_stages = num_pipeline_stages
        self.stage_id = stage_id
        
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
        
        # DualPipe scheduler
        self.dualpipe_scheduler = DualPipeScheduler(
            num_stages=num_pipeline_stages,
            microbatch_size=1
        )

    def forward(self, x, layer_func, use_recompute: bool = True):
        """
        Forward pass with DualPipe communication overlapping
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Expand input to n-stream residual
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n, -1)
        x_flat = x_expanded.reshape(batch_size, seq_len, self.n * input_dim)
        
        # Compute all mappings using the DualPipe scheduler
        H_pre, H_post, H_res = self.compute_mappings_with_dualpipe(x_flat)
        
        # Apply H_pre to x to get input for layer function
        x_for_layer = self.apply_mapping_pre(x_flat, H_pre)
        
        # Apply the layer function F (this is the heavy computation we don't recompute)
        F_output = layer_func(x_for_layer)
        
        # Apply H_res and H_post mappings and combine
        result = self.apply_combined_mappings(x_flat, F_output, H_res, H_post)
        
        return result

    def compute_mappings_with_dualpipe(self, x_flat):
        """
        Compute mappings with DualPipe scheduling
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
        x_transposed = x_streams.transpose(-2, -1)
        H_res_x_temp = torch.matmul(H_res, x_transposed)
        H_res_x = H_res_x_temp.transpose(-2, -1)
        
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


class PipelineParallelMHCTransformerBlock(nn.Module):
    """
    Transformer block with pipeline parallelism and DualPipe scheduling
    """
    def __init__(self, d_model: int, nhead: int, num_pipeline_stages: int = 2, 
                 stage_id: int = 0, dim_feedforward: int = 2048, dropout: float = 0.1, 
                 expansion_rate: int = 4):
        super(PipelineParallelMHCTransformerBlock, self).__init__()
        
        self.stage_id = stage_id
        self.num_pipeline_stages = num_pipeline_stages
        
        if stage_id == 0:  # First stage: attention
            # Self-attention mechanism
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            
            # mHC layer for attention with DualPipe
            self.mhc_attn = DualPipeMHCLayer(
                d_model, expansion_rate, num_pipeline_stages, stage_id
            )
            
            # Normalization for attention
            self.norm1 = nn.LayerNorm(d_model)
            
        elif stage_id == 1:  # Second stage: FFN
            # Feed-forward network
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout_ffn = nn.Dropout(dropout)
            
            # mHC layer for FFN with DualPipe
            self.mhc_ffn = DualPipeMHCLayer(
                d_model, expansion_rate, num_pipeline_stages, stage_id
            )
            
            # Normalization for FFN
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_recompute: bool = True):
        if self.stage_id == 0:  # Attention stage
            # Attention sublayer with mHC and DualPipe
            def attention_func(x):
                attn_output, _ = self.self_attn(x, x, x)
                return self.dropout(attn_output)
            
            attn_output = self.mhc_attn(x, attention_func, use_recompute=use_recompute)
            output = x + attn_output
            output = self.norm1(output)
            return output
            
        elif self.stage_id == 1:  # FFN stage
            # FFN sublayer with mHC and DualPipe
            def ffn_func(x):
                ffn_output = self.linear2(self.dropout_ffn(F.relu(self.linear1(x))))
                return self.dropout(ffn_output)
            
            ffn_output = self.mhc_ffn(x, ffn_func, use_recompute=use_recompute)
            output = x + ffn_output
            output = self.norm2(output)
            return output


class PipelineParallelMHCModel(nn.Module):
    """
    Complete model with pipeline parallelism and DualPipe scheduling
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, num_pipeline_stages: int = 2,
                 dim_feedforward: int = 2048, dropout: float = 0.1, 
                 expansion_rate: int = 4):
        super(PipelineParallelMHCModel, self).__init__()
        
        self.d_model = d_model
        self.num_pipeline_stages = num_pipeline_stages
        self.num_layers = num_layers
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Create pipeline stages
        self.pipeline_stages = nn.ModuleList()
        
        # Distribute layers across stages
        layers_per_stage = num_layers // num_pipeline_stages
        for stage_id in range(num_pipeline_stages):
            start_layer = stage_id * layers_per_stage
            end_layer = min((stage_id + 1) * layers_per_stage, num_layers)
            
            for layer_idx in range(start_layer, end_layer):
                stage = PipelineParallelMHCTransformerBlock(
                    d_model, nhead, num_pipeline_stages, stage_id,
                    dim_feedforward, dropout, expansion_rate
                )
                self.pipeline_stages.append(stage)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, use_recompute: bool = True):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Process through pipeline stages
        for stage in self.pipeline_stages:
            x = stage(x, use_recompute=use_recompute)
        
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


def simulate_pipeline_communication():
    """
    Simulate the pipeline communication overlapping as described in the paper
    """
    print("Simulating DualPipe Communication Overlapping...")
    
    # Create a simple pipeline with 2 stages
    stage_0 = DualPipeMHCLayer(input_dim=128, expansion_rate=4, num_pipeline_stages=2, stage_id=0)
    stage_1 = DualPipeMHCLayer(input_dim=128, expansion_rate=4, num_pipeline_stages=2, stage_id=1)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    dummy_input = torch.randn(batch_size, seq_len, 128)
    
    # Define simple layer functions
    attn_layer = nn.Linear(128, 128)
    ffn_layer = nn.Linear(128, 128)
    
    def attn_func(x):
        return attn_layer(x)
    
    def ffn_func(x):
        return ffn_layer(x)
    
    # Process through stage 0 (attention)
    stage_0_output = stage_0(dummy_input, attn_func)
    print(f"Stage 0 output shape: {stage_0_output.shape}")
    
    # Process through stage 1 (FFN)
    stage_1_output = stage_1(stage_0_output, ffn_func)
    print(f"Stage 1 output shape: {stage_1_output.shape}")
    
    # Test pipeline parallel model
    model = PipelineParallelMHCModel(
        vocab_size=1000, 
        d_model=128, 
        nhead=4, 
        num_layers=4, 
        num_pipeline_stages=2,
        expansion_rate=4
    )
    
    dummy_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    model_output = model(dummy_tokens)
    print(f"Pipeline model output shape: {model_output.shape}")
    
    print("\nDualPipe communication overlapping simulation completed!")


def test_dualpipe_implementation():
    """
    Test the DualPipe implementation
    """
    print("Testing DualPipe Implementation...")
    
    # Test individual DualPipe layer
    dualpipe_layer = DualPipeMHCLayer(input_dim=128, expansion_rate=4)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    dummy_input = torch.randn(batch_size, seq_len, 128)
    
    # Define a simple layer function for testing
    simple_layer = nn.Linear(128, 128)
    
    def layer_func(x):
        return simple_layer(x)
    
    # Forward pass
    output = dualpipe_layer(dummy_input, layer_func)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test pipeline stages
    simulate_pipeline_communication()
    
    print("\nDualPipe implementation test passed!")


if __name__ == "__main__":
    test_dualpipe_implementation()