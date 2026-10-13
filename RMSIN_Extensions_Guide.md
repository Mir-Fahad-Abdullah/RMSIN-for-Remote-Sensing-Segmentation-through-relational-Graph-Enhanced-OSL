# RMSIN Extensions: Comprehensive Technical Guide

## Project Context

This guide is tailored to your RMSIN (Rotated Multi-Scale Interaction Network) implementation for Referring Remote Sensing Image Segmentation (RRSIS) on the RRSIS-D dataset. All guidance references the existing codebase: Swin-B + BERT backbone, PWAM fusion, SimpleDecoding with ARC, and your standalone GNN fusion modules (GCNFusion, CrossModalGNN, GraphSAGEFusion).

---

## 1. Relational Graph-Enhanced Module

### 1.1 Conceptual Design

The goal is to model **relational structure** between visual regions and linguistic tokens beyond what standard cross-attention (PWAM) captures. In RRSIS, spatial relationships matter enormously — expressions like *"the vehicle on the lower right of the bridge"* require reasoning about inter-object relations.

Your existing GNN modules (Cell 50–52) are a strong starting point. Here's how to think about the design choices:

**Node Representations:**

There are three strategies for constructing graph nodes from the Swin backbone features:

| Strategy | Nodes | Pros | Cons |
|----------|-------|------|------|
| **Pixel/Token-level** | Each spatial position H×W becomes a node | Fine-grained; matches your current PWAM interface | O(N²) edges; expensive for 480×480 images |
| **Region-level** | Superpixel or pooled region features | Semantically meaningful; tractable graph size | Requires region extraction step |
| **Patch-level** | Non-overlapping patches (e.g., 8×8 or 16×16 grid) | Good balance; aligned with Swin's windowed attention | Loses some fine-grained detail |

**Recommendation for Colab free-tier**: Use patch-level nodes. At stage 3 of Swin-B (feature map ~15×15 for 480px input), you already have ~225 tokens. This is tractable for GNN message passing. Your current GCNFusion already operates at this level — that's the right call.

**Edge Construction:**

Your GCNFusion uses **feature-similarity adjacency** (top-k cosine similarity). Consider adding:

```python
# Spatial adjacency: connect tokens within a spatial neighborhood
def build_spatial_adj(H, W, radius=2):
    """Connect each token to its spatial neighbors within radius."""
    N = H * W
    coords = torch.stack(torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).reshape(N, 2).float()
    dist = torch.cdist(coords, coords)  # (N, N)
    adj = (dist <= radius).float()
    return adj

# Hybrid adjacency: combine spatial + feature similarity
def build_hybrid_adj(x, H, W, k=8, radius=2, alpha=0.5):
    """
    x: (B, N, C) visual features
    Returns: (B, N, N) normalized adjacency
    """
    B, N, C = x.shape
    # Feature similarity
    xn = F.normalize(x, dim=-1)
    sim = torch.bmm(xn, xn.transpose(1, 2))
    topk_v, topk_i = sim.topk(k, dim=-1)
    feat_adj = torch.zeros(B, N, N, device=x.device)
    feat_adj.scatter_(-1, topk_i, topk_v)
    
    # Spatial adjacency (same for all batches)
    spat_adj = build_spatial_adj(H, W, radius).to(x.device)
    spat_adj = spat_adj.unsqueeze(0).expand(B, -1, -1)
    
    # Combine
    adj = alpha * feat_adj + (1 - alpha) * spat_adj
    deg = adj.sum(-1, keepdim=True).clamp(min=1e-6)
    return adj / deg
```

**Language-Conditioned Graph Attention:**

Instead of a fixed adjacency, you can make edge weights language-aware:

```python
class LanguageConditionedGraphAttention(nn.Module):
    """
    Edge weights are modulated by linguistic context.
    This allows the graph to attend differently based on
    what the expression is describing.
    """
    def __init__(self, dim, l_dim=768):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.lang_gate = nn.Sequential(
            nn.Linear(l_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.scale = dim ** -0.5
    
    def forward(self, x, lang_ctx):
        """
        x: (B, N, C) visual node features
        lang_ctx: (B, C) pooled language context
        """
        q = self.query(x)
        k = self.key(x)
        
        # Language-conditioned gating on queries
        gate = self.lang_gate(lang_ctx).unsqueeze(1)  # (B, 1, C)
        q = q * gate
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, x)
```

### 1.2 Integration Point

Your notebook (Cell 54) already identifies the integration point: replace `self.fusion = PWAM(...)` in `MMBasicLayer.__init__` with any GNN module. The gating in `MMBasicLayer.forward()` stays the same.

**Recommended integration approach:**

```python
# In MMBasicLayer.__init__:
if fusion_type == 'gcn':
    self.fusion = GCNFusion(dim, dim, 768, dim, dim, num_heads, fusion_drop)
elif fusion_type == 'cross_modal_gnn':
    self.fusion = CrossModalGNN(dim, dim, 768, dim, dim, num_heads, fusion_drop)
elif fusion_type == 'graphsage':
    self.fusion = GraphSAGEFusion(dim, dim, 768, dim, dim, num_heads, fusion_drop)
else:
    self.fusion = PWAM(dim, dim, 768, dim, dim, num_heads, fusion_drop)
```

Add `fusion_type` as a config parameter for easy ablation.

### 1.3 Advanced: Multi-Hop Relational Reasoning

For complex spatial expressions (e.g., *"the baseball field on the right of the green ground track field in the middle"*), consider multi-hop reasoning:

```python
class MultiHopRelationalModule(nn.Module):
    """
    Performs K hops of message passing, where each hop
    integrates different relational information.
    """
    def __init__(self, dim, num_hops=3, l_dim=768):
        super().__init__()
        self.hops = num_hops
        self.msg_layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_hops)
        ])
        self.update_layers = nn.ModuleList([
            nn.GRUCell(dim, dim) for _ in range(num_hops)
        ])
        self.lang_proj = nn.Linear(l_dim, dim)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_hops)])
    
    def forward(self, x, adj, lang_pool):
        """
        x: (B, N, C)
        adj: (B, N, N) normalized adjacency
        lang_pool: (B, C) pooled language vector
        """
        B, N, C = x.shape
        h = x
        lang_bias = self.lang_proj(lang_pool).unsqueeze(1)  # (B, 1, C)
        
        for hop in range(self.hops):
            # Aggregate neighbor messages
            msg = self.msg_layers[hop](torch.bmm(adj, h))
            msg = msg + lang_bias  # language-conditioned
            
            # GRU update (flatten for GRUCell)
            h_flat = h.reshape(B * N, C)
            msg_flat = msg.reshape(B * N, C)
            h_flat = self.update_layers[hop](msg_flat, h_flat)
            h = self.norms[hop](h_flat.reshape(B, N, C))
        
        return h
```

---

## 2. Open-Set Network Design

### 2.1 The Problem

In standard RRSIS, the model only sees categories present in RRSIS-D (20 categories: Airplane, Ship, Vehicle, Bridge, etc.). An open-set extension means the model should:
- Correctly segment known classes when the referring expression matches
- Detect and reject queries about **unseen/unknown** classes (e.g., a class not in RRSIS-D)
- Produce calibrated confidence to distinguish "I found it" from "this doesn't exist in the image"

### 2.2 Strategy 1: Confidence-Based Thresholding

The simplest approach. Your model outputs 2-class logits (background vs. foreground). The maximum softmax probability indicates confidence:

```python
class OpenSetSegmentationHead(nn.Module):
    """
    Wraps the decoder output with open-set rejection.
    If max confidence is below threshold, predict all-background.
    """
    def __init__(self, threshold=0.65):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, logits):
        """logits: (B, 2, H, W)"""
        probs = F.softmax(logits, dim=1)
        fg_prob = probs[:, 1, :, :]  # foreground probability
        
        # Per-image maximum foreground confidence
        max_conf = fg_prob.view(fg_prob.size(0), -1).max(dim=1)[0]  # (B,)
        
        # Reject: set all foreground logits to -inf for low-confidence samples
        reject_mask = (max_conf < self.threshold).view(-1, 1, 1, 1)
        logits = torch.where(reject_mask.expand_as(logits),
                            torch.tensor([10.0, -10.0], device=logits.device).view(1, 2, 1, 1).expand_as(logits),
                            logits)
        return logits, max_conf
```

### 2.3 Strategy 2: Energy-Based OOD Detection

Energy scores provide better separation than softmax for out-of-distribution detection:

```python
class EnergyBasedOOD(nn.Module):
    """
    Uses energy score E(x) = -T * log(sum(exp(f_i(x)/T)))
    Lower energy = more in-distribution.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.T = temperature
    
    def compute_energy(self, logits):
        """logits: (B, 2, H, W) → per-pixel energy"""
        return -self.T * torch.logsumexp(logits / self.T, dim=1)  # (B, H, W)
    
    def forward(self, logits, energy_threshold=-5.0):
        energy = self.compute_energy(logits)  # (B, H, W)
        
        # Average energy per image
        avg_energy = energy.mean(dim=(1, 2))  # (B,)
        
        # High energy (more positive) = OOD
        is_ood = avg_energy > energy_threshold
        
        return {
            'energy_map': energy,
            'avg_energy': avg_energy,
            'is_ood': is_ood
        }
```

### 2.4 Strategy 3: Text-Visual Alignment Score

This leverages the cross-modal nature of RRSIS. If the text description doesn't match anything in the image, the alignment score should be low:

```python
class TextVisualAlignmentGate(nn.Module):
    """
    Computes a global text-visual alignment score.
    Low alignment → expression likely refers to unseen content.
    """
    def __init__(self, v_dim, l_dim=768, hidden=256):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hidden)
        self.l_proj = nn.Linear(l_dim, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visual_features, lang_features, l_mask):
        """
        visual_features: (B, C, H, W) - deepest encoder features
        lang_features: (B, 768, N_l)
        l_mask: (B, N_l, 1)
        """
        # Pool visual
        v_pool = visual_features.flatten(2).mean(dim=2)  # (B, C)
        v = self.v_proj(v_pool)  # (B, hidden)
        
        # Pool language (masked mean)
        l = lang_features.permute(0, 2, 1)  # (B, N_l, 768)
        mask = l_mask.float()
        l_pool = (l * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 768)
        l = self.l_proj(l_pool)  # (B, hidden)
        
        # Alignment score via element-wise product
        alignment = self.classifier(v * l)  # (B, 1)
        return alignment.squeeze(1)
```

### 2.5 Training for Open-Set

During training, you can synthesize "unknown" samples by:
1. **Random text shuffle**: Pair images with mismatched expressions → model should predict all-background
2. **Category holdout**: Hold out 2–3 categories during training, use them as "unknown" during validation
3. **Auxiliary loss**: Add a binary classification head predicting "does this expression match anything in the image?"

```python
# In training loop, with probability p, replace the expression with a random one
if random.random() < 0.15:  # 15% of samples get mismatched text
    # Swap with expression from a different image
    shuffled_idx = torch.randperm(sentences.size(0))
    sentences = sentences[shuffled_idx]
    attentions = attentions[shuffled_idx]
    target = torch.zeros_like(target)  # all background
    is_matched = torch.zeros(B, device=device)
else:
    is_matched = torch.ones(B, device=device)
```

---

## 3. Knowledge Distillation Strategy

### 3.1 Why KD for RRSIS?

Knowledge distillation lets you:
- Transfer knowledge from a large teacher (e.g., RMSIN with Swin-B) to a smaller student (e.g., RMSIN with Swin-T) for deployment
- Distill from an ensemble of models for better single-model performance
- Transfer cross-modal alignment knowledge from foundation models (CLIP, SAM) into your specialized RRSIS model

### 3.2 Teacher–Student Architecture

```
Teacher: RMSIN (Swin-B + BERT-base) — pretrained, frozen
Student: RMSIN (Swin-T + BERT-small) or RMSIN with different fusion module
```

```python
class KnowledgeDistillationWrapper(nn.Module):
    """
    Wraps teacher and student for KD training.
    Teacher is frozen; only student parameters are updated.
    """
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.T = temperature
        self.alpha = alpha  # weight for KD loss vs task loss
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
    
    def forward(self, image, sentences, attentions, target=None):
        # Teacher forward (no grad)
        with torch.no_grad():
            t_logits = self.teacher(image, sentences, attentions)
            if t_logits.shape[-2:] != target.shape[-2:]:
                t_logits = F.interpolate(t_logits, target.shape[-2:],
                                         mode='bilinear', align_corners=False)
        
        # Student forward
        s_logits = self.student(image, sentences, attentions)
        if s_logits.shape[-2:] != target.shape[-2:]:
            s_logits = F.interpolate(s_logits, target.shape[-2:],
                                     mode='bilinear', align_corners=False)
        
        return s_logits, t_logits
```

### 3.3 Distillation Loss Functions

**Output-level KD (soft targets):**

```python
class OutputDistillationLoss(nn.Module):
    """KL divergence between softened teacher and student outputs."""
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature
    
    def forward(self, s_logits, t_logits):
        """Both: (B, 2, H, W)"""
        s_soft = F.log_softmax(s_logits / self.T, dim=1)
        t_soft = F.softmax(t_logits / self.T, dim=1)
        
        # KL divergence, scaled by T^2 as per Hinton et al.
        kd_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.T ** 2)
        return kd_loss
```

**Feature-level KD (intermediate feature mimicry):**

```python
class FeatureDistillationLoss(nn.Module):
    """
    Aligns intermediate features between teacher and student.
    Requires hooking into both models to extract features.
    """
    def __init__(self, teacher_dims, student_dims):
        super().__init__()
        # Projection layers to match dimensions
        self.projectors = nn.ModuleList([
            nn.Conv2d(s_dim, t_dim, 1) if s_dim != t_dim else nn.Identity()
            for s_dim, t_dim in zip(student_dims, teacher_dims)
        ])
    
    def forward(self, s_features, t_features):
        """
        s_features, t_features: lists of (B, C_i, H_i, W_i)
        """
        loss = 0.0
        for proj, sf, tf in zip(self.projectors, s_features, t_features):
            sf_proj = proj(sf)
            if sf_proj.shape[-2:] != tf.shape[-2:]:
                sf_proj = F.interpolate(sf_proj, tf.shape[-2:],
                                        mode='bilinear', align_corners=False)
            loss += F.mse_loss(sf_proj, tf)
        return loss / len(s_features)
```

**Attention Transfer:**

```python
class AttentionTransferLoss(nn.Module):
    """
    Transfers attention maps (spatial activation patterns)
    from teacher to student. From Zagoruyko & Komodakis (2017).
    """
    def forward(self, s_features, t_features):
        loss = 0.0
        for sf, tf in zip(s_features, t_features):
            # Attention map: channel-wise L2 norm, then normalize
            s_attn = F.normalize(sf.pow(2).mean(1).view(sf.size(0), -1), dim=1)
            t_attn = F.normalize(tf.pow(2).mean(1).view(tf.size(0), -1), dim=1)
            loss += (s_attn - t_attn).pow(2).mean()
        return loss / len(s_features)
```

### 3.4 Combined Training Objective

```python
class DistillationTrainingLoss(nn.Module):
    def __init__(self, task_loss, temperature=4.0,
                 alpha_kd=0.3, alpha_feat=0.1, alpha_attn=0.05):
        super().__init__()
        self.task_loss = task_loss  # Your existing CombinedLoss
        self.kd_loss = OutputDistillationLoss(temperature)
        self.feat_loss = FeatureDistillationLoss(
            teacher_dims=[128, 256, 512, 1024],  # Swin-B
            student_dims=[96, 192, 384, 768]      # Swin-T
        )
        self.attn_loss = AttentionTransferLoss()
        self.alpha_kd = alpha_kd
        self.alpha_feat = alpha_feat
        self.alpha_attn = alpha_attn
    
    def forward(self, s_logits, t_logits, target,
                s_features=None, t_features=None):
        # Task loss (CE + Dice on ground truth)
        L_task = self.task_loss(s_logits, target)
        
        # KD loss (soft targets from teacher)
        L_kd = self.kd_loss(s_logits, t_logits)
        
        total = L_task + self.alpha_kd * L_kd
        
        if s_features is not None and t_features is not None:
            L_feat = self.feat_loss(s_features, t_features)
            L_attn = self.attn_loss(s_features, t_features)
            total += self.alpha_feat * L_feat + self.alpha_attn * L_attn
        
        return total
```

### 3.5 Foundation Model Distillation (CLIP/SAM → RMSIN)

You can also distill knowledge from CLIP or SAM:

- **CLIP distillation**: Use CLIP's visual encoder as teacher for the Swin backbone. Align Swin features to CLIP visual features via MSE loss. This transfers CLIP's strong vision-language alignment.
- **SAM distillation**: Use SAM's mask decoder outputs as soft segmentation targets. Since RRSIS-D was partially built using SAM, this creates a natural consistency.

```python
# CLIP feature alignment (conceptual)
clip_visual = clip_model.encode_image(image)  # (B, 512)
swin_pool = F.adaptive_avg_pool2d(swin_features[-1], 1).flatten(1)  # (B, 1024)
swin_proj = clip_projector(swin_pool)  # (B, 512)
clip_align_loss = F.mse_loss(swin_proj, clip_visual.detach())
```

---

## 4. Custom Segmentation Loss Design

### 4.1 Current Loss

Your notebook (Cell 31) uses `CombinedLoss = 0.9 * CE + 0.1 * Dice`. This is a good baseline. Here's how to extend it.

### 4.2 Dice Loss (already implemented)

Your implementation is correct. One improvement — use a per-sample Dice rather than batch-level:

```python
class ImprovedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """pred: (B, 2, H, W), target: (B, H, W) long"""
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # foreground prob
        target_f = target.float()
        
        # Per-sample Dice
        inter = (pred_soft * target_f).sum(dim=(1, 2))
        union = pred_soft.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()
```

### 4.3 Boundary Loss

Boundary loss focuses the model on segmentation edges, which is critical for RRSIS where objects have complex boundaries (especially rotated objects):

```python
class BoundaryLoss(nn.Module):
    """
    Penalizes incorrect predictions near ground-truth boundaries.
    Uses Laplacian edge detection on the GT mask.
    """
    def __init__(self, theta=3):
        super().__init__()
        self.theta = theta
        # Laplacian kernel for edge detection
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', kernel)
    
    def get_boundary(self, mask):
        """Extract boundary from binary mask using Laplacian."""
        mask_f = mask.float().unsqueeze(1)  # (B, 1, H, W)
        edges = F.conv2d(mask_f, self.laplacian, padding=1)
        edges = (edges.abs() > 0).float().squeeze(1)  # (B, H, W)
        
        # Dilate boundary by theta pixels
        if self.theta > 1:
            kernel_size = 2 * self.theta + 1
            edges = F.max_pool2d(edges.unsqueeze(1), kernel_size,
                                stride=1, padding=self.theta).squeeze(1)
        return edges
    
    def forward(self, pred, target):
        """
        pred: (B, 2, H, W) logits
        target: (B, H, W) binary mask
        """
        boundary = self.get_boundary(target)  # (B, H, W)
        
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # foreground prob
        
        # Weighted BCE on boundary pixels
        boundary_pred = pred_soft * boundary
        boundary_gt = target.float() * boundary
        
        # Only compute loss where boundary exists
        num_boundary = boundary.sum().clamp(min=1)
        loss = F.binary_cross_entropy(
            boundary_pred.clamp(1e-6, 1-1e-6),
            boundary_gt,
            reduction='sum'
        ) / num_boundary
        
        return loss
```

### 4.4 Rotation Consistency Loss

This is specific to RRSIS and leverages the ARC module's rotation awareness:

```python
class RotationConsistencyLoss(nn.Module):
    """
    Enforces that segmentation predictions are consistent
    under small rotations of the input.
    
    For a rotated input, the prediction should be the same
    rotation of the original prediction.
    """
    def __init__(self, max_angle=15):
        super().__init__()
        self.max_angle = max_angle
    
    def forward(self, model, image, sentences, attentions):
        """
        Run model twice: once on original, once on rotated input.
        Compare predictions for consistency.
        """
        # Original prediction
        with torch.no_grad():
            pred_orig = model(image, sentences, attentions)
        
        # Random rotation
        angle = (torch.rand(1).item() * 2 - 1) * self.max_angle  # [-max, +max]
        angle_rad = angle * 3.14159 / 180.0
        
        # Rotation matrix for affine_grid
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        B = image.size(0)
        theta = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a, cos_a, 0]], dtype=image.dtype,
                             device=image.device).unsqueeze(0).expand(B, -1, -1)
        
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        image_rot = F.grid_sample(image, grid, align_corners=False)
        
        # Prediction on rotated input
        pred_rot = model(image_rot, sentences, attentions)
        
        # Rotate original prediction to match
        if pred_orig.shape[-2:] != image.shape[-2:]:
            pred_orig = F.interpolate(pred_orig, image.shape[-2:],
                                      mode='bilinear', align_corners=False)
        if pred_rot.shape[-2:] != image.shape[-2:]:
            pred_rot = F.interpolate(pred_rot, image.shape[-2:],
                                     mode='bilinear', align_corners=False)
        
        pred_orig_rot = F.grid_sample(pred_orig, grid, align_corners=False)
        
        # Consistency loss: predictions should match
        loss = F.mse_loss(F.softmax(pred_rot, dim=1),
                          F.softmax(pred_orig_rot.detach(), dim=1))
        return loss
```

**Important note**: Rotation consistency loss requires an extra forward pass, which doubles memory. On Colab free tier, consider:
- Applying it only every N batches (e.g., every 5th batch)
- Using smaller rotation angles (±5° instead of ±15°)
- Applying it only during later epochs when the model is more stable

### 4.5 Combined Loss with Balancing

```python
class FullCompositeLoss(nn.Module):
    """
    L_total = w_ce * L_ce + w_dice * L_dice + w_boundary * L_boundary
    
    Rotation consistency loss is computed separately because it
    requires a second forward pass.
    """
    def __init__(self, w_ce=0.5, w_dice=0.3, w_boundary=0.2,
                 ce_class_weights=None):
        super().__init__()
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        
        if ce_class_weights is not None:
            self.register_buffer('ce_weight', torch.FloatTensor(ce_class_weights))
        else:
            self.register_buffer('ce_weight', torch.FloatTensor([0.9, 1.1]))
        
        self.dice = ImprovedDiceLoss()
        self.boundary = BoundaryLoss(theta=3)
    
    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, target.shape[-2:],
                                 mode='bilinear', align_corners=False)
        
        target_long = target.squeeze(1).long()
        
        L_ce = F.cross_entropy(pred, target_long, weight=self.ce_weight)
        L_dice = self.dice(pred, target_long)
        L_bnd = self.boundary(pred, target_long)
        
        total = self.w_ce * L_ce + self.w_dice * L_dice + self.w_boundary * L_bnd
        return total
```

**Loss weight schedule**: Consider increasing boundary loss weight over training:

```python
# In training loop:
epoch_frac = epoch / total_epochs
boundary_weight = 0.1 + 0.2 * epoch_frac  # ramp from 0.1 to 0.3
```

---

## 5. Handling Class Imbalance (Rare Classes)

### 5.1 The RRSIS-D Imbalance Problem

From the paper (Fig. 3), RRSIS-D has severe imbalance: "Airplane" is ~15.6% of all samples while categories like "Dam," "Golf field," and "Chimney" have far fewer samples. Also, from Fig. 4, most targets are tiny masks (θ < 0.1), meaning foreground pixels are rare relative to background.

### 5.2 Strategy 1: Class-Aware CE Weights

Your current CE weights are [0.9, 1.1] (bg, fg). These can be computed dynamically:

```python
def compute_class_weights(dataset, num_classes=2):
    """
    Compute inverse-frequency weights from training set.
    """
    pixel_counts = torch.zeros(num_classes)
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        for c in range(num_classes):
            pixel_counts[c] += (mask == c).sum()
    
    total = pixel_counts.sum()
    weights = total / (num_classes * pixel_counts)
    return weights / weights.sum() * num_classes  # normalize

# Typical result for RRSIS-D: ~[0.55, 1.45] (bg much more frequent)
```

### 5.3 Strategy 2: Focal Loss

Focal loss down-weights easy examples and focuses on hard pixels:

```python
class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2 works well for most segmentation tasks.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.FloatTensor(alpha))
        else:
            self.alpha = None
    
    def forward(self, pred, target):
        """pred: (B, 2, H, W), target: (B, H, W) long"""
        ce = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        
        if self.alpha is not None:
            alpha_t = self.alpha[target]  # (B, H, W)
            focal = alpha_t * focal
        
        return focal.mean()
```

### 5.4 Strategy 3: Category-Level Sampling

Balance the dataloader so rare categories appear more often:

```python
class CategoryBalancedSampler(torch.utils.data.Sampler):
    """
    Oversamples rare categories so each category appears
    roughly equally during training.
    """
    def __init__(self, dataset, category_labels, num_samples=None):
        self.indices = list(range(len(dataset)))
        self.num_samples = num_samples or len(dataset)
        
        # Count per category
        label_counts = Counter(category_labels)
        
        # Weight = inverse frequency
        weights = [1.0 / label_counts[category_labels[i]]
                   for i in self.indices]
        self.weights = torch.DoubleTensor(weights)
    
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples,
                                      replacement=True).tolist())
    
    def __len__(self):
        return self.num_samples
```

### 5.5 Strategy 4: Copy-Paste Augmentation for Rare Objects

```python
def copy_paste_augmentation(image, mask, rare_image, rare_mask):
    """
    Paste a rare-class object onto a training image.
    Both image and mask are augmented.
    """
    # Find bounding box of rare object
    ys, xs = torch.where(rare_mask > 0)
    if len(ys) == 0:
        return image, mask
    
    y1, y2 = ys.min().item(), ys.max().item()
    x1, x2 = xs.min().item(), xs.max().item()
    
    # Crop the object
    obj_img = rare_image[:, y1:y2+1, x1:x2+1]
    obj_mask = rare_mask[y1:y2+1, x1:x2+1]
    
    # Random placement
    h, w = obj_mask.shape
    H, W = mask.shape
    if h >= H or w >= W:
        return image, mask
    
    py = random.randint(0, H - h)
    px = random.randint(0, W - w)
    
    # Paste
    paste_mask = obj_mask.bool()
    image[:, py:py+h, px:px+w][:, paste_mask] = obj_img[:, paste_mask]
    mask[py:py+h, px:px+w][paste_mask] = obj_mask[paste_mask]
    
    return image, mask
```

### 5.6 Strategy 5: Online Hard Example Mining (OHEM)

```python
class OHEMLoss(nn.Module):
    """Keep only the top-k% hardest pixels for loss computation."""
    def __init__(self, hard_ratio=0.3):
        super().__init__()
        self.hard_ratio = hard_ratio
    
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)
        
        # Keep top hard_ratio of pixels
        B = ce.size(0)
        ce_flat = ce.view(B, -1)
        k = max(1, int(ce_flat.size(1) * self.hard_ratio))
        topk_loss, _ = ce_flat.topk(k, dim=1)
        
        return topk_loss.mean()
```

### 5.7 Recommended Combination

```python
# For RRSIS-D, I recommend:
loss = FullCompositeLoss(
    w_ce=0.0,      # Replace CE with Focal
    w_dice=0.3,
    w_boundary=0.2,
)
focal = FocalLoss(alpha=[0.6, 1.4], gamma=2.0)

# In training:
L_focal = focal(pred, target)
L_composite = loss(pred, target)
total_loss = 0.5 * L_focal + L_composite
```

---

## 6. Explainable AI (XAI) for RRSIS

### 6.1 Grad-CAM / Grad-CAM++

The most natural choice. Visualize which spatial regions the model attends to for a given expression:

```python
class GradCAM:
    """
    Generates Grad-CAM heatmaps for the RMSIN model.
    Target layer: typically the last encoder stage (F_o^4).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, image, sentences, attentions, target_class=1):
        self.model.eval()
        output = self.model(image, sentences, attentions)
        
        # Backprop from target class score
        self.model.zero_grad()
        if output.shape[-2:] != image.shape[-2:]:
            output = F.interpolate(output, image.shape[-2:],
                                   mode='bilinear', align_corners=False)
        score = output[:, target_class].sum()
        score.backward()
        
        # Weighted combination of activation maps
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to input size
        cam = F.interpolate(cam, image.shape[-2:],
                            mode='bilinear', align_corners=False)
        return cam.squeeze(1)
```

### 6.2 Cross-Modal Attention Visualization

Since RMSIN fuses vision and language, visualize the cross-attention maps from the PWAM/GNN fusion modules:

```python
class AttentionVisualizer:
    """
    Hooks into the cross-modal attention layers to extract
    and visualize attention maps.
    """
    def __init__(self, model):
        self.attention_maps = {}
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.MultiheadAttention,)):
                module.register_forward_hook(
                    lambda m, i, o, n=name: self._save_attn(n, o))
    
    def _save_attn(self, name, output):
        # MultiheadAttention returns (attn_output, attn_weights)
        if len(output) > 1 and output[1] is not None:
            self.attention_maps[name] = output[1].detach()
    
    def visualize(self, image, attn_map, word_tokens, save_path=None):
        """
        attn_map: (N_visual, N_lang) attention weights
        word_tokens: list of word strings
        """
        import matplotlib.pyplot as plt
        
        H = W = int(attn_map.shape[0] ** 0.5)
        
        fig, axes = plt.subplots(1, min(len(word_tokens), 6),
                                 figsize=(3 * min(len(word_tokens), 6), 3))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, (ax, word) in enumerate(zip(axes, word_tokens)):
            word_attn = attn_map[:, idx].reshape(H, W)
            word_attn = F.interpolate(
                word_attn.unsqueeze(0).unsqueeze(0),
                image.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            ax.imshow(word_attn, alpha=0.5, cmap='jet')
            ax.set_title(word)
            ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
```

### 6.3 GNN Explainability (for your Graph-Enhanced Module)

If you use GCNFusion or CrossModalGNN, visualize the learned adjacency/attention:

```python
class GNNExplainer:
    """
    Visualizes the graph structure learned by the GNN fusion module.
    Shows which visual tokens are connected and how strongly.
    """
    @staticmethod
    def visualize_adjacency(adj_matrix, H, W, image, top_node=None):
        """
        adj_matrix: (N, N) adjacency from GCN
        H, W: spatial dimensions of the feature map
        image: original image for overlay
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if top_node is None:
            # Pick the node with highest degree
            top_node = adj_matrix.sum(dim=1).argmax().item()
        
        # Get neighbors of top_node
        neighbors = adj_matrix[top_node]
        neighbor_map = neighbors.reshape(H, W)
        
        # Upsample to image size
        neighbor_map = F.interpolate(
            neighbor_map.unsqueeze(0).unsqueeze(0).float(),
            image.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax1.set_title('Input Image')
        
        ax2.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax2.imshow(neighbor_map, alpha=0.6, cmap='hot')
        ax2.set_title(f'Graph Neighbors of Node {top_node}')
        
        plt.tight_layout()
        plt.show()
```

### 6.4 SHAP-style Occlusion Sensitivity

Test which image regions and which words in the expression are most important:

```python
def text_occlusion_sensitivity(model, image, sentences, attentions, tokenizer):
    """
    Mask each word one at a time and measure change in prediction.
    Returns importance score per word.
    """
    model.eval()
    with torch.no_grad():
        base_pred = model(image, sentences, attentions)
        base_pred = F.softmax(base_pred, dim=1)[:, 1]
        base_score = base_pred.sum().item()
    
    token_ids = sentences[0].cpu().tolist()
    words = tokenizer.convert_ids_to_tokens(token_ids)
    mask_token_id = tokenizer.mask_token_id
    
    importances = []
    for i in range(len(token_ids)):
        if token_ids[i] in [tokenizer.cls_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id]:
            importances.append(0.0)
            continue
        
        masked = sentences.clone()
        masked[0, i] = mask_token_id
        
        with torch.no_grad():
            masked_pred = model(image, masked, attentions)
            masked_pred = F.softmax(masked_pred, dim=1)[:, 1]
            masked_score = masked_pred.sum().item()
        
        importances.append(abs(base_score - masked_score))
    
    return words, importances
```

### 6.5 Recommendation Summary for XAI

| Method | What It Explains | Complexity | Recommended? |
|--------|-----------------|------------|--------------|
| **Grad-CAM** | Which spatial regions matter | Low | Yes — primary |
| **Cross-Modal Attention** | Word-to-region alignment | Low | Yes — primary |
| **GNN Adjacency** | Graph structure | Medium | Yes, if using GNN |
| **Text Occlusion** | Word importance | Medium | Yes — for qualitative |
| **SHAP (full)** | Comprehensive | Very High | No — too expensive for Colab |

---

## 7. Evaluation Metrics

### 7.1 Core Metrics (Already in Your Notebook)

Your `SegmentationMetrics` class (Cell 34) already computes the standard RRSIS metrics from the paper:

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **oIoU** | Σ Intersection / Σ Union (global) | Overall segmentation quality |
| **mIoU** | Mean of per-image IoU | Average per-sample quality |
| **Pr@K** | Fraction with IoU ≥ K | Precision at thresholds {0.5, 0.6, 0.7, 0.8, 0.9} |

These match exactly what RMSIN reports in Table 2 of the paper. Keep using these.

### 7.2 Additional Metrics You Should Add

**F1 Score (per-pixel):**

```python
class PixelF1:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
    
    def update(self, pred_mask, gt_mask):
        pred = (pred_mask > 0.5).float().view(-1)
        gt = (gt_mask > 0.5).float().view(-1)
        self.tp += (pred * gt).sum().item()
        self.fp += (pred * (1 - gt)).sum().item()
        self.fn += ((1 - pred) * gt).sum().item()
    
    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {'precision': precision, 'recall': recall, 'F1': f1}
```

**Boundary IoU (BIoU):**

Measures segmentation quality specifically at object boundaries — very relevant for rotated objects:

```python
class BoundaryIoU:
    """
    Computes IoU only within a narrow band around GT boundaries.
    More sensitive to boundary accuracy than standard IoU.
    """
    def __init__(self, dilation=3):
        self.dilation = dilation
        self.reset()
    
    def reset(self):
        self._biou_scores = []
    
    def _get_boundary(self, mask, dilation):
        kernel = torch.ones(1, 1, 2*dilation+1, 2*dilation+1,
                           device=mask.device)
        dilated = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0),
                          kernel, padding=dilation)
        eroded = -F.conv2d(-mask.float().unsqueeze(0).unsqueeze(0),
                           kernel, padding=dilation)
        boundary = ((dilated > 0) & (eroded <= 0)).float().squeeze()
        return boundary
    
    def update(self, pred_mask, gt_mask):
        """Both: (B, H, W) binary"""
        B = pred_mask.size(0)
        for b in range(B):
            gt_bnd = self._get_boundary(gt_mask[b], self.dilation)
            pred_bnd = self._get_boundary(pred_mask[b], self.dilation)
            
            # IoU within boundary region
            boundary_region = ((gt_bnd + pred_bnd) > 0).float()
            if boundary_region.sum() == 0:
                self._biou_scores.append(1.0)
                continue
            
            pred_in_bnd = pred_mask[b].float() * boundary_region
            gt_in_bnd = gt_mask[b].float() * boundary_region
            
            inter = (pred_in_bnd * gt_in_bnd).sum()
            union = (pred_in_bnd + gt_in_bnd).clamp(max=1).sum()
            self._biou_scores.append((inter / union.clamp(min=1e-6)).item())
    
    def compute(self):
        return {'BIoU': np.mean(self._biou_scores)}
```

**Scale-Stratified Metrics:**

Since RRSIS-D has huge scale variation, report metrics per scale bucket:

```python
class ScaleStratifiedMetrics:
    """
    Reports mIoU separately for small, medium, and large objects.
    Scale is measured as mask_area / image_area.
    """
    def __init__(self, thresholds=(0.01, 0.1)):
        self.small_t, self.large_t = thresholds
        self.reset()
    
    def reset(self):
        self.small_ious = []
        self.medium_ious = []
        self.large_ious = []
    
    def update(self, pred_mask, gt_mask):
        """Both: (B, H, W)"""
        B = pred_mask.size(0)
        for b in range(B):
            gt_area = gt_mask[b].float().mean().item()
            
            inter = (pred_mask[b] * gt_mask[b]).sum().float()
            union = (pred_mask[b] + gt_mask[b]).clamp(max=1).sum().float()
            iou = (inter / union.clamp(min=1e-6)).item()
            
            if gt_area < self.small_t:
                self.small_ious.append(iou)
            elif gt_area < self.large_t:
                self.medium_ious.append(iou)
            else:
                self.large_ious.append(iou)
    
    def compute(self):
        return {
            'mIoU_small': np.mean(self.small_ious) if self.small_ious else 0,
            'mIoU_medium': np.mean(self.medium_ious) if self.medium_ious else 0,
            'mIoU_large': np.mean(self.large_ious) if self.large_ious else 0,
            'count_small': len(self.small_ious),
            'count_medium': len(self.medium_ious),
            'count_large': len(self.large_ious),
        }
```

**Category-wise mIoU:**

```python
class CategoryMetrics:
    """Track mIoU per RRSIS-D category (if category labels available)."""
    def __init__(self, categories):
        self.categories = categories
        self.reset()
    
    def reset(self):
        self.cat_ious = {cat: [] for cat in self.categories}
    
    def update(self, pred_mask, gt_mask, category):
        inter = (pred_mask * gt_mask).sum().float()
        union = (pred_mask + gt_mask).clamp(max=1).sum().float()
        iou = (inter / union.clamp(min=1e-6)).item()
        self.cat_ious[category].append(iou)
    
    def compute(self):
        results = {}
        for cat, ious in self.cat_ious.items():
            if ious:
                results[f'mIoU_{cat}'] = np.mean(ious)
        return results
```

### 7.3 Open-Set Specific Metrics

If you implement open-set detection (Section 2), add:

```python
class OpenSetMetrics:
    """
    AUROC: Area under ROC for known vs unknown detection
    FPR@95: False positive rate when true positive rate = 95%
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.scores = []
        self.labels = []  # 1 = known, 0 = unknown
    
    def update(self, confidence_score, is_known):
        self.scores.extend(confidence_score.cpu().tolist())
        self.labels.extend(is_known.cpu().tolist())
    
    def compute(self):
        from sklearn.metrics import roc_auc_score, roc_curve
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        auroc = roc_auc_score(labels, scores)
        
        fpr, tpr, _ = roc_curve(labels, scores)
        idx = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[idx]
        
        return {'AUROC': auroc, 'FPR@95': fpr95}
```

### 7.4 KD-Specific Metrics

When evaluating knowledge distillation, report:

| Metric | Purpose |
|--------|---------|
| **Student mIoU** vs **Teacher mIoU** | How close is the student? |
| **Compression ratio** | Parameters_teacher / Parameters_student |
| **Speedup** | FPS_student / FPS_teacher |
| **Feature correlation** | Pearson correlation between teacher and student intermediate features |

```python
def model_efficiency_metrics(model, input_size=(1, 3, 480, 480), device='cuda'):
    """Compute FLOPs, params, and inference speed."""
    from torchinfo import summary
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Inference speed
    model.eval()
    dummy_img = torch.randn(*input_size).to(device)
    dummy_sent = torch.ones(1, 20, dtype=torch.long).to(device)
    dummy_attn = torch.ones(1, 20, dtype=torch.long).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_img, dummy_sent, dummy_attn)
    
    torch.cuda.synchronize()
    import time
    start = time.time()
    N = 50
    for _ in range(N):
        with torch.no_grad():
            _ = model(dummy_img, dummy_sent, dummy_attn)
    torch.cuda.synchronize()
    fps = N / (time.time() - start)
    
    return {'params_M': params, 'FPS': fps}
```

### 7.5 Complete Evaluation Suite

```python
class FullEvaluationSuite:
    """Aggregates all metrics for comprehensive evaluation."""
    def __init__(self, categories=None):
        self.seg_metrics = SegmentationMetrics()  # Your existing class
        self.pixel_f1 = PixelF1()
        self.boundary_iou = BoundaryIoU()
        self.scale_metrics = ScaleStratifiedMetrics()
        self.cat_metrics = CategoryMetrics(categories) if categories else None
    
    def reset(self):
        self.seg_metrics.reset()
        self.pixel_f1.reset()
        self.boundary_iou.reset()
        self.scale_metrics.reset()
        if self.cat_metrics:
            self.cat_metrics.reset()
    
    def update(self, pred_logits, gt_mask, category=None):
        pred_mask = (pred_logits.argmax(dim=1) > 0)
        gt_bin = (gt_mask.squeeze(1) > 0)
        
        self.seg_metrics.update(pred_mask, gt_bin)
        self.pixel_f1.update(pred_mask, gt_bin)
        self.boundary_iou.update(pred_mask, gt_bin)
        self.scale_metrics.update(pred_mask, gt_bin)
        if self.cat_metrics and category:
            for b, cat in enumerate(category):
                self.cat_metrics.update(pred_mask[b], gt_bin[b], cat)
    
    def compute(self):
        results = {}
        results.update(self.seg_metrics.compute())
        results.update(self.pixel_f1.compute())
        results.update(self.boundary_iou.compute())
        results.update(self.scale_metrics.compute())
        if self.cat_metrics:
            results.update(self.cat_metrics.compute())
        return results
```

---

## Summary: What to Prioritize

Given your Colab free-tier constraints and CSE465 project scope, here's a prioritized implementation order:

1. **GNN Fusion Integration** (Section 1) — You already have the modules. Integrate CrossModalGNN first, benchmark against PWAM.

2. **Composite Loss** (Section 4) — Replace your current CombinedLoss with Focal + Dice + Boundary. This will give immediate mIoU gains.

3. **Evaluation Metrics** (Section 7) — Add BIoU, scale-stratified metrics, and F1. These strengthen your paper's analysis.

4. **Class Imbalance** (Section 5) — Add Focal loss and category-balanced sampling. Critical for rare categories.

5. **XAI** (Section 6) — Grad-CAM + cross-modal attention visualization for qualitative results.

6. **Knowledge Distillation** (Section 3) — If time permits, train a Swin-T student from your Swin-B teacher.

7. **Open-Set** (Section 2) — Most experimental; implement the text-visual alignment gate + mismatched training if time permits.
