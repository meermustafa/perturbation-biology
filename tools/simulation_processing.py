import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.metrics import adjusted_rand_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# 1. SIMULATE PERTURB-SEQ DATA
# =============================================================================

def simulate_perturbseq_data(n_cells=2000, n_genes=5000, n_guide_types=20):
    """
    Simulate realistic Perturb-seq data with:
    - Control cells (no perturbation)
    - Various sgRNA perturbations affecting different gene modules
    """
    print("Simulating Perturb-seq data...")
    
    # Define cell types and their proportions
    guide_names = ['Control'] + [f'sgRNA_{i}' for i in range(1, n_guide_types)]
    
    # Assign cells to perturbations (more controls than perturbed)
    cell_assignments = np.random.choice(
        guide_names, 
        size=n_cells, 
        p=[0.4] + [0.6/(n_guide_types-1)]*(n_guide_types-1)
    )
    
    # Create baseline gene expression (log-normal distribution)
    baseline_expression = np.random.lognormal(mean=1.0, sigma=1.5, size=(n_cells, n_genes))
    
    # Add technical noise
    technical_noise = np.random.normal(0, 0.1, size=(n_cells, n_genes))
    
    # Create gene modules (groups of co-regulated genes)
    n_modules = 10
    module_size = n_genes // n_modules
    
    # Simulate perturbation effects
    perturbed_expression = baseline_expression.copy()
    
    for i, guide in enumerate(guide_names[1:], 1):  # Skip control
        # Each sgRNA affects 1-2 gene modules
        affected_modules = np.random.choice(n_modules, size=np.random.randint(1, 3), replace=False)
        
        cells_with_guide = cell_assignments == guide
        
        for module in affected_modules:
            module_start = module * module_size
            module_end = min((module + 1) * module_size, n_genes)
            
            # Random effect size and direction
            effect_size = np.random.uniform(-2, 2)
            module_effect = np.random.normal(effect_size, 0.3, size=module_end-module_start)
            
            perturbed_expression[cells_with_guide, module_start:module_end] *= np.exp(module_effect)
    
    # Add technical noise and ensure non-negative values
    final_expression = perturbed_expression + technical_noise
    final_expression = np.maximum(final_expression, 0.01)
    
    # Log transform (common for scRNA-seq)
    log_expression = np.log1p(final_expression)
    
    # Create metadata
    metadata = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_cells)],
        'guide': cell_assignments,
        'is_control': cell_assignments == 'Control'
    })
    
    # Create gene names
    gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    print(f"Created dataset: {n_cells} cells × {n_genes} genes")
    print(f"Perturbations: {len(guide_names)} types")
    print(f"Control cells: {sum(cell_assignments == 'Control')}")
    
    return log_expression, metadata, gene_names

# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def preprocess_data(expression_data, n_hvgs=2000):
    """
    Standard scRNA-seq preprocessing:
    - Feature selection (highly variable genes)
    - Normalization
    """
    print("Preprocessing data...")
    
    # Calculate gene statistics
    gene_means = np.mean(expression_data, axis=0)
    gene_vars = np.var(expression_data, axis=0)
    gene_cv = gene_vars / (gene_means + 1e-8)
    
    # Select highly variable genes
    hvg_indices = np.argsort(gene_cv)[-n_hvgs:]
    hvg_data = expression_data[:, hvg_indices]
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(hvg_data)
    
    print(f"Selected {n_hvgs} highly variable genes")
    
    return scaled_data, hvg_indices, scaler

# =============================================================================
# 3. AUTOENCODER MODEL
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)

# =============================================================================
# 4. VARIATIONAL AUTOENCODER MODEL
# =============================================================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss = Reconstruction loss + KL divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# =============================================================================
# 5. TRAINING FUNCTIONS
# =============================================================================

def train_autoencoder(model, train_loader, epochs=100, lr=0.001):
    """Train standard autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data[0].to(device)  # Unpack tuple from TensorDataset
            
            optimizer.zero_grad()
            reconstructed, _ = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        
        if epoch % 20 == 0:
            print(f'AE Epoch {epoch}, Loss: {losses[-1]:.4f}')
    
    return losses

def train_vae(model, train_loader, epochs=100, lr=0.001, beta=1.0):
    """Train variational autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch_data in train_loader:
            batch_data = batch_data[0].to(device)  # Unpack tuple from TensorDataset
            
            optimizer.zero_grad()
            reconstructed, mu, logvar, _ = model(batch_data)
            loss, recon_loss, kl_loss = vae_loss_function(
                reconstructed, batch_data, mu, logvar, beta
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        recon_losses.append(epoch_recon / len(train_loader))
        kl_losses.append(epoch_kl / len(train_loader))
        
        if epoch % 20 == 0:
            print(f'VAE Epoch {epoch}, Total Loss: {losses[-1]:.4f}, '
                  f'Recon: {recon_losses[-1]:.4f}, KL: {kl_losses[-1]:.4f}')
    
    return losses, recon_losses, kl_losses

# =============================================================================
# 6. VALIDATION AND COMPARISON FUNCTIONS
# =============================================================================

def evaluate_reconstruction_quality(model, data_loader, model_type='AE'):
    """Evaluate how well models reconstruct the original data"""
    model.eval()
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data[0].to(device)
            
            if model_type == 'AE':
                reconstructed, _ = model(batch_data)
            else:  # VAE
                reconstructed, _, _, _ = model(batch_data)
            
            mse = F.mse_loss(reconstructed, batch_data, reduction='sum')
            total_mse += mse.item()
            total_samples += batch_data.size(0)
    
    return total_mse / total_samples

def calculate_latent_space_metrics(embeddings, metadata):
    """Calculate metrics for latent space quality"""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    
    # Silhouette score (higher = better separation)
    guide_labels = metadata['guide'].values
    sil_score = silhouette_score(embeddings, guide_labels)
    
    # Local neighborhood preservation
    # How well does the embedding preserve local structure?
    nbrs = NearestNeighbors(n_neighbors=10).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Calculate average purity of neighborhoods
    neighborhood_purities = []
    for i, neighbors in enumerate(indices):
        cell_guide = guide_labels[i]
        neighbor_guides = guide_labels[neighbors]
        purity = np.mean(neighbor_guides == cell_guide)
        neighborhood_purities.append(purity)
    
    avg_neighborhood_purity = np.mean(neighborhood_purities)
    
    return {
        'silhouette_score': sil_score,
        'neighborhood_purity': avg_neighborhood_purity
    }
def perform_differential_expression_validation(processed_data, embeddings, metadata, original_data, hvg_indices):
    """
    Use differential expression as ground truth to validate perturbation detection
    This simulates using known biology to validate our representations
    """
    from scipy import stats
    
    print("Validating against simulated ground truth differential expression...")
    
    # For each perturbation, calculate true DE effect sizes
    control_mask = metadata['is_control'].values
    control_expression = original_data[control_mask]  # Use original full data
    
    true_effects = {}
    latent_effects = {}
    
    for guide in metadata['guide'].unique():
        if guide == 'Control':
            continue
            
        guide_mask = metadata['guide'] == guide
        guide_expression = original_data[guide_mask]  # Use original full data
        
        # True effect: differential expression (fold changes)
        control_mean = np.mean(control_expression, axis=0)
        guide_mean = np.mean(guide_expression, axis=0)
        
        # Focus on HVGs used in the model (now we have the right dimensions)
        control_hvg = control_mean[hvg_indices]
        guide_hvg = guide_mean[hvg_indices]
        
        # Calculate log fold changes
        log_fc = np.log2((guide_hvg + 1e-6) / (control_hvg + 1e-6))
        true_effect_size = np.mean(np.abs(log_fc))
        true_effects[guide] = true_effect_size
        
        # Latent effect: distance in embedding space
        control_embeddings = embeddings[control_mask]
        guide_embeddings = embeddings[guide_mask]
        
        control_centroid = np.mean(control_embeddings, axis=0)
        guide_centroid = np.mean(guide_embeddings, axis=0)
        
        latent_distance = np.linalg.norm(guide_centroid - control_centroid)
        latent_effects[guide] = latent_distance
    
    # Calculate correlation between true and latent effects
    guides = list(true_effects.keys())
    true_vals = [true_effects[g] for g in guides]
    latent_vals = [latent_effects[g] for g in guides]
    
    correlation = np.corrcoef(true_vals, latent_vals)[0, 1]
    
    return correlation, true_effects, latent_effects


def test_generative_capability(vae_model, original_data, n_samples=100):
    """Test VAE's ability to generate realistic samples"""
    vae_model.eval()
    
    with torch.no_grad():
        # Sample from prior distribution
        latent_dim = vae_model.fc_mu.out_features
        z_samples = torch.randn(n_samples, latent_dim).to(device)
        generated = vae_model.decode(z_samples)
        generated_np = generated.cpu().numpy()
    
    # Compare statistics of generated vs real data
    original_mean = np.mean(original_data, axis=0)
    generated_mean = np.mean(generated_np, axis=0)
    
    original_std = np.std(original_data, axis=0)
    generated_std = np.std(generated_np, axis=0)
    
    mean_similarity = np.corrcoef(original_mean, generated_mean)[0, 1]
    std_similarity = np.corrcoef(original_std, generated_std)[0, 1]
    
    return {
        'mean_correlation': mean_similarity,
        'std_correlation': std_similarity,
        'generated_samples': generated_np
    }

def cross_validation_stability(processed_data, metadata, n_splits=5):
    """Test how stable the representations are across different train/test splits"""
    from sklearn.model_selection import KFold
    
    print("Testing representation stability across CV folds...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    input_dim = processed_data.shape[1]
    latent_dim = 50
    
    ae_correlations = []
    vae_correlations = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(processed_data)):
        print(f"  Fold {fold + 1}/{n_splits}")
        
        # Create train/validation data
        train_data = processed_data[train_idx]
        val_data = processed_data[val_idx]
        val_metadata = metadata.iloc[val_idx].copy()
        
        # Train models on this fold
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        val_tensor = torch.FloatTensor(val_data)
        val_dataset = torch.utils.data.TensorDataset(val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Initialize and train models
        ae_fold = Autoencoder(input_dim, latent_dim).to(device)
        vae_fold = VAE(input_dim, latent_dim).to(device)
        
        train_autoencoder(ae_fold, train_loader, epochs=50)  # Fewer epochs for speed
        train_vae(vae_fold, train_loader, epochs=50)
        
        # Get embeddings
        ae_embeddings = get_embeddings(ae_fold, val_loader, 'AE')
        vae_embeddings = get_embeddings(vae_fold, val_loader, 'VAE')
        
        # Calculate ground truth correlation for this fold
        # Calculate ground truth correlation for this fold
        ae_corr, _, _ = perform_differential_expression_validation(
            val_data, ae_embeddings, val_metadata, val_data, np.arange(input_dim)
        )
        vae_corr, _, _ = perform_differential_expression_validation(
            val_data, vae_embeddings, val_metadata, val_data, np.arange(input_dim)
        )
        
        ae_correlations.append(ae_corr)
        vae_correlations.append(vae_corr)
    
    return {
        'ae_correlations': ae_correlations,
        'vae_correlations': vae_correlations,
        'ae_mean': np.mean(ae_correlations),
        'ae_std': np.std(ae_correlations),
        'vae_mean': np.mean(vae_correlations),
        'vae_std': np.std(vae_correlations)
    }

# =============================================================================
# 6. ANALYSIS FUNCTIONS
# =============================================================================

def get_embeddings(model, data_loader, model_type='AE'):
    """Extract latent embeddings from trained model"""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data[0].to(device)  # Unpack tuple from TensorDataset
            
            if model_type == 'AE':
                _, latent = model(batch_data)
                embeddings.append(latent.cpu().numpy())
            else:  # VAE
                mu, logvar = model.encode(batch_data)
                embeddings.append(mu.cpu().numpy())  # Use mean for embedding
    
    return np.vstack(embeddings)

def analyze_perturbation_effects(embeddings, metadata):
    """Analyze perturbation effects in latent space"""
    print("\nAnalyzing perturbation effects...")
    
    # Calculate distances from control centroid
    control_mask = metadata['is_control'].values
    control_embeddings = embeddings[control_mask]
    control_centroid = np.mean(control_embeddings, axis=0)
    
    # Distance of each cell from control centroid
    distances = np.linalg.norm(embeddings - control_centroid, axis=1)
    
    # Group by perturbation
    results = []
    for guide in metadata['guide'].unique():
        if guide == 'Control':
            continue
            
        guide_mask = metadata['guide'] == guide
        guide_distances = distances[guide_mask]
        control_distances = distances[control_mask]
        
        # Statistical test
        statistic, p_value = stats.mannwhitneyu(
            guide_distances, control_distances, alternative='greater'
        )
        
        results.append({
            'guide': guide,
            'n_cells': sum(guide_mask),
            'mean_distance': np.mean(guide_distances),
            'control_mean_distance': np.mean(control_distances),
            'fold_change': np.mean(guide_distances) / np.mean(control_distances),
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    results_df = pd.DataFrame(results)
    results_df['fdr'] = stats.false_discovery_control(results_df['p_value'])
    
    return results_df, distances

# =============================================================================
# 7. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_curves(ae_losses, vae_losses, vae_recon_losses, vae_kl_losses):
    """Plot training curves comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # AE loss
    axes[0,0].plot(ae_losses, 'b-', label='AE Loss')
    axes[0,0].set_title('Autoencoder Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].grid(True, alpha=0.3)
    
    # VAE total loss
    axes[0,1].plot(vae_losses, 'r-', label='VAE Total Loss')
    axes[0,1].set_title('VAE Training Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Total Loss')
    axes[0,1].grid(True, alpha=0.3)
    
    # VAE components
    axes[1,0].plot(vae_recon_losses, 'g-', label='Reconstruction')
    axes[1,0].plot(vae_kl_losses, 'm-', label='KL Divergence')
    axes[1,0].set_title('VAE Loss Components')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Comparison
    # Normalize losses for comparison
    ae_norm = np.array(ae_losses) / ae_losses[0]
    vae_norm = np.array(vae_losses) / vae_losses[0]
    
    axes[1,1].plot(ae_norm, 'b-', label='AE (normalized)', linewidth=2)
    axes[1,1].plot(vae_norm, 'r-', label='VAE (normalized)', linewidth=2)
    axes[1,1].set_title('Training Convergence Comparison')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Normalized Loss')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_embeddings_comparison(ae_embeddings, vae_embeddings, metadata):
    """Compare AE vs VAE embeddings using UMAP"""
    print("Computing UMAP projections...")
    
    # Compute UMAP for both
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    ae_umap = umap_reducer.fit_transform(ae_embeddings)
    
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    vae_umap = umap_reducer.fit_transform(vae_embeddings)
    
    # Create color map for guides
    unique_guides = metadata['guide'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_guides)))
    color_map = dict(zip(unique_guides, colors))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # AE UMAP
    for guide in unique_guides:
        mask = metadata['guide'] == guide
        axes[0].scatter(ae_umap[mask, 0], ae_umap[mask, 1], 
                       c=[color_map[guide]], label=guide, alpha=0.7, s=20)
    
    axes[0].set_title('Autoencoder Latent Space (UMAP)')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # VAE UMAP
    for guide in unique_guides:
        mask = metadata['guide'] == guide
        axes[1].scatter(vae_umap[mask, 0], vae_umap[mask, 1], 
                       c=[color_map[guide]], label=guide, alpha=0.7, s=20)
    
    axes[1].set_title('VAE Latent Space (UMAP)')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ae_umap, vae_umap

def plot_perturbation_analysis(ae_results, vae_results):
    """Compare perturbation effect detection between AE and VAE"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Effect sizes comparison
    guides = ae_results['guide']
    ae_effects = ae_results['fold_change']
    vae_effects = vae_results['fold_change']
    
    x = np.arange(len(guides))
    width = 0.35
    
    bars1 = axes[0,0].bar(x - width/2, ae_effects, width, label='AE', alpha=0.8)
    bars2 = axes[0,0].bar(x + width/2, vae_effects, width, label='VAE', alpha=0.8)
    
    axes[0,0].set_xlabel('sgRNA')
    axes[0,0].set_ylabel('Fold Change in Distance from Control')
    axes[0,0].set_title('Perturbation Effect Size Comparison')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(guides, rotation=45, ha='right')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # P-values comparison
    ae_pvals = -np.log10(ae_results['p_value'] + 1e-10)
    vae_pvals = -np.log10(vae_results['p_value'] + 1e-10)
    
    bars1 = axes[0,1].bar(x - width/2, ae_pvals, width, label='AE', alpha=0.8)
    bars2 = axes[0,1].bar(x + width/2, vae_pvals, width, label='VAE', alpha=0.8)
    
    axes[0,1].set_xlabel('sgRNA')
    axes[0,1].set_ylabel('-log10(p-value)')
    axes[0,1].set_title('Statistical Significance Comparison')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(guides, rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                      alpha=0.7, label='p=0.05')
    
    # Correlation between methods
    axes[1,0].scatter(ae_effects, vae_effects, alpha=0.7)
    axes[1,0].set_xlabel('AE Fold Change')
    axes[1,0].set_ylabel('VAE Fold Change')
    axes[1,0].set_title('Effect Size Correlation (AE vs VAE)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(ae_effects, vae_effects)[0,1]
    axes[1,0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1,0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Detection sensitivity
    ae_detected = sum(ae_results['significant'])
    vae_detected = sum(vae_results['significant'])
    total_perturbations = len(guides)
    
    methods = ['Autoencoder', 'VAE']
    detected = [ae_detected, vae_detected]
    
    bars = axes[1,1].bar(methods, detected, alpha=0.8, 
                        color=['blue', 'red'])
    axes[1,1].set_ylabel('Number of Significant Perturbations')
    axes[1,1].set_title('Detection Sensitivity Comparison')
    axes[1,1].set_ylim(0, total_perturbations)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add numbers on bars
    for bar, count in zip(bars, detected):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{count}/{total_perturbations}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return corr

def plot_comprehensive_validation(ae_metrics, vae_metrics, cv_results, 
                                ae_generative=None, vae_generative=None):
    """Plot comprehensive validation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Reconstruction quality
    methods = ['Autoencoder', 'VAE']
    recon_scores = [ae_metrics['reconstruction_mse'], vae_metrics['reconstruction_mse']]
    
    bars = axes[0,0].bar(methods, recon_scores, color=['blue', 'red'], alpha=0.7)
    axes[0,0].set_ylabel('Mean Squared Error')
    axes[0,0].set_title('Reconstruction Quality\n(Lower = Better)')
    axes[0,0].grid(True, alpha=0.3)
    
    for bar, score in zip(bars, recon_scores):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Latent space quality metrics
    sil_scores = [ae_metrics['latent_metrics']['silhouette_score'], 
                  vae_metrics['latent_metrics']['silhouette_score']]
    
    bars = axes[0,1].bar(methods, sil_scores, color=['blue', 'red'], alpha=0.7)
    axes[0,1].set_ylabel('Silhouette Score')
    axes[0,1].set_title('Cluster Separation\n(Higher = Better)')
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars, sil_scores):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
    
    # 3. Ground truth correlation
    gt_correlations = [ae_metrics['ground_truth_correlation'], 
                       vae_metrics['ground_truth_correlation']]
    
    bars = axes[0,2].bar(methods, gt_correlations, color=['blue', 'red'], alpha=0.7)
    axes[0,2].set_ylabel('Correlation with True DE')
    axes[0,2].set_title('Biological Validity\n(Higher = Better)')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim(0, 1)
    
    for bar, score in zip(bars, gt_correlations):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
    
    # 4. Cross-validation stability
    ae_cv = cv_results['ae_correlations']
    vae_cv = cv_results['vae_correlations']
    
    bp = axes[1,0].boxplot([ae_cv, vae_cv], labels=methods, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    axes[1,0].set_ylabel('Ground Truth Correlation')
    axes[1,0].set_title('Cross-Validation Stability')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add mean values
    axes[1,0].text(1, cv_results['ae_mean'] + 0.05, 
                  f'μ={cv_results["ae_mean"]:.3f}', ha='center')
    axes[1,0].text(2, cv_results['vae_mean'] + 0.05, 
                  f'μ={cv_results["vae_mean"]:.3f}', ha='center')
    
    # 5. Neighborhood purity
    purity_scores = [ae_metrics['latent_metrics']['neighborhood_purity'], 
                     vae_metrics['latent_metrics']['neighborhood_purity']]
    
    bars = axes[1,1].bar(methods, purity_scores, color=['blue', 'red'], alpha=0.7)
    axes[1,1].set_ylabel('Neighborhood Purity')
    axes[1,1].set_title('Local Structure Preservation\n(Higher = Better)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(0, 1)
    
    for bar, score in zip(bars, purity_scores):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
    
    # 6. Generative quality (VAE only)
    if vae_generative:
        gen_metrics = ['Mean Correlation', 'Std Correlation']
        gen_scores = [vae_generative['mean_correlation'], vae_generative['std_correlation']]
        
        bars = axes[1,2].bar(gen_metrics, gen_scores, color='red', alpha=0.7)
        axes[1,2].set_ylabel('Correlation with Real Data')
        axes[1,2].set_title('VAE Generative Quality\n(Higher = Better)')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].set_ylim(0, 1)
        
        for bar, score in zip(bars, gen_scores):
            axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom')
        
        axes[1,2].text(0.5, 0.1, 'AE cannot generate\nnew samples', 
                      transform=axes[1,2].transAxes, ha='center',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def create_validation_summary_table(ae_metrics, vae_metrics, cv_results):
    """Create a summary table comparing all metrics"""
    
    summary_data = {
        'Metric': [
            'Reconstruction MSE',
            'Silhouette Score',
            'Neighborhood Purity', 
            'Ground Truth Correlation',
            'CV Stability (mean)',
            'CV Stability (std)',
            'Overall Score'
        ],
        'Autoencoder': [
            f"{ae_metrics['reconstruction_mse']:.4f}",
            f"{ae_metrics['latent_metrics']['silhouette_score']:.4f}",
            f"{ae_metrics['latent_metrics']['neighborhood_purity']:.4f}",
            f"{ae_metrics['ground_truth_correlation']:.4f}",
            f"{cv_results['ae_mean']:.4f}",
            f"{cv_results['ae_std']:.4f}",
            "TBD"
        ],
        'VAE': [
            f"{vae_metrics['reconstruction_mse']:.4f}",
            f"{vae_metrics['latent_metrics']['silhouette_score']:.4f}",
            f"{vae_metrics['latent_metrics']['neighborhood_purity']:.4f}",
            f"{vae_metrics['ground_truth_correlation']:.4f}",
            f"{cv_results['vae_mean']:.4f}",
            f"{cv_results['vae_std']:.4f}",
            "TBD"
        ],
        'Winner': ['', '', '', '', '', '', '']
    }
    
    # Determine winners (lower is better for reconstruction MSE and CV std)
    ae_recon = ae_metrics['reconstruction_mse']
    vae_recon = vae_metrics['reconstruction_mse']
    summary_data['Winner'][0] = 'AE' if ae_recon < vae_recon else 'VAE'
    
    ae_sil = ae_metrics['latent_metrics']['silhouette_score']
    vae_sil = vae_metrics['latent_metrics']['silhouette_score']
    summary_data['Winner'][1] = 'AE' if ae_sil > vae_sil else 'VAE'
    
    ae_purity = ae_metrics['latent_metrics']['neighborhood_purity']
    vae_purity = vae_metrics['latent_metrics']['neighborhood_purity']
    summary_data['Winner'][2] = 'AE' if ae_purity > vae_purity else 'VAE'
    
    ae_gt = ae_metrics['ground_truth_correlation']
    vae_gt = vae_metrics['ground_truth_correlation']
    summary_data['Winner'][3] = 'AE' if ae_gt > vae_gt else 'VAE'
    
    summary_data['Winner'][4] = 'AE' if cv_results['ae_mean'] > cv_results['vae_mean'] else 'VAE'
    summary_data['Winner'][5] = 'AE' if cv_results['ae_std'] < cv_results['vae_std'] else 'VAE'
    
    # Calculate overall winner
    ae_wins = summary_data['Winner'][:6].count('AE')
    vae_wins = summary_data['Winner'][:6].count('VAE')
    summary_data['Winner'][6] = 'AE' if ae_wins > vae_wins else 'VAE' if vae_wins > ae_wins else 'TIE'
    summary_data['Autoencoder'][6] = f"{ae_wins}/6 wins"
    summary_data['VAE'][6] = f"{vae_wins}/6 wins"
    
    return pd.DataFrame(summary_data)


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

def main():
    print("=== Comprehensive Perturb-seq Analysis: Autoencoder vs VAE ===\n")
    
    # Generate data
    expression_data, metadata, gene_names = simulate_perturbseq_data()
    
    # Preprocess
    processed_data, hvg_indices, scaler = preprocess_data(expression_data)
    
    # Create data loaders
    tensor_data = torch.FloatTensor(processed_data)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    input_dim = processed_data.shape[1]
    latent_dim = 50
    
    print(f"\nModel specifications:")
    print(f"Input dimension: {input_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Training on: {device}")
    
    # Initialize models
    ae_model = Autoencoder(input_dim, latent_dim).to(device)
    vae_model = VAE(input_dim, latent_dim).to(device)
    
    print(f"\nAutoencoder parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    print(f"VAE parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    
    # Train models
    print("\n=== Training Models ===")
    ae_losses = train_autoencoder(ae_model, train_loader, epochs=100)
    vae_losses, vae_recon_losses, vae_kl_losses = train_vae(vae_model, train_loader, epochs=100)
    
    # Extract embeddings
    print("\n=== Extracting Embeddings ===")
    ae_embeddings = get_embeddings(ae_model, eval_loader, 'AE')
    vae_embeddings = get_embeddings(vae_model, eval_loader, 'VAE')
    
    print(f"AE embeddings shape: {ae_embeddings.shape}")
    print(f"VAE embeddings shape: {vae_embeddings.shape}")
    
    # =============================================================================
    # COMPREHENSIVE VALIDATION
    # =============================================================================
    
    print("\n=== COMPREHENSIVE VALIDATION ===")
    
    # 1. Reconstruction Quality
    print("1. Evaluating reconstruction quality...")
    ae_recon_mse = evaluate_reconstruction_quality(ae_model, eval_loader, 'AE')
    vae_recon_mse = evaluate_reconstruction_quality(vae_model, eval_loader, 'VAE')
    
    # 2. Latent Space Quality
    print("2. Calculating latent space metrics...")
    ae_latent_metrics = calculate_latent_space_metrics(ae_embeddings, metadata)
    vae_latent_metrics = calculate_latent_space_metrics(vae_embeddings, metadata)
    
    # 3. Ground Truth Validation
    print("3. Validating against ground truth differential expression...")
    ae_gt_corr, ae_true_effects, ae_latent_effects = perform_differential_expression_validation(
        processed_data, ae_embeddings, metadata, expression_data, hvg_indices
    )
    vae_gt_corr, vae_true_effects, vae_latent_effects = perform_differential_expression_validation(
        processed_data, vae_embeddings, metadata, expression_data, hvg_indices
    )
    
    # 4. Generative Quality (VAE only)
    print("4. Testing generative capabilities...")
    vae_generative = test_generative_capability(vae_model, processed_data)
    
    # 5. Cross-validation stability
    print("5. Testing cross-validation stability...")
    cv_results = cross_validation_stability(processed_data, metadata, n_splits=3)  # Reduced for speed
    
    # Compile all metrics
    ae_metrics = {
        'reconstruction_mse': ae_recon_mse,
        'latent_metrics': ae_latent_metrics,
        'ground_truth_correlation': ae_gt_corr
    }
    
    vae_metrics = {
        'reconstruction_mse': vae_recon_mse,
        'latent_metrics': vae_latent_metrics,
        'ground_truth_correlation': vae_gt_corr
    }
    
    # =============================================================================
    # ORIGINAL ANALYSES
    # =============================================================================
    
    # Analyze perturbation effects
    print("\n=== Analyzing Perturbation Effects ===")
    ae_results, ae_distances = analyze_perturbation_effects(ae_embeddings, metadata)
    vae_results, vae_distances = analyze_perturbation_effects(vae_embeddings, metadata)
    
    print("\nAutoencoder Results:")
    print(ae_results[['guide', 'fold_change', 'p_value', 'significant']].round(4))
    
    print("\nVAE Results:")
    print(vae_results[['guide', 'fold_change', 'p_value', 'significant']].round(4))
    
    # =============================================================================
    # COMPREHENSIVE VISUALIZATIONS
    # =============================================================================
    
    print("\n=== Generating Comprehensive Visualizations ===")
    
    # Original plots
    plot_training_curves(ae_losses, vae_losses, vae_recon_losses, vae_kl_losses)
    ae_umap, vae_umap = plot_embeddings_comparison(ae_embeddings, vae_embeddings, metadata)
    correlation = plot_perturbation_analysis(ae_results, vae_results)
    
    # New validation plots
    plot_comprehensive_validation(ae_metrics, vae_metrics, cv_results, 
                                vae_generative=vae_generative)
    
    # =============================================================================
    # FINAL SUMMARY AND RECOMMENDATIONS
    # =============================================================================
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("="*80)
    
    # Create and display summary table
    summary_table = create_validation_summary_table(ae_metrics, vae_metrics, cv_results)
    print("\nDETAILED COMPARISON:")
    print(summary_table.to_string(index=False))
    
    print(f"\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    # Determine overall winner based on multiple criteria
    scores = {
        'ae_reconstruction': 1 if ae_recon_mse < vae_recon_mse else 0,
        'ae_silhouette': 1 if ae_latent_metrics['silhouette_score'] > vae_latent_metrics['silhouette_score'] else 0,
        'ae_purity': 1 if ae_latent_metrics['neighborhood_purity'] > vae_latent_metrics['neighborhood_purity'] else 0,
        'ae_ground_truth': 1 if ae_gt_corr > vae_gt_corr else 0,
        'ae_stability': 1 if cv_results['ae_std'] < cv_results['vae_std'] else 0,
        'ae_bio_validity': 1 if cv_results['ae_mean'] > cv_results['vae_mean'] else 0
    }
    
    ae_total_score = sum(scores.values())
    vae_total_score = 6 - ae_total_score
    
    print(f"\nOVERALL SCORES:")
    print(f"Autoencoder: {ae_total_score}/6 metrics")
    print(f"VAE: {vae_total_score}/6 metrics")
    
    if ae_total_score > vae_total_score:
        winner = "AUTOENCODER"
        print(f"\n🏆 WINNER: {winner}")
        print("\nThe Autoencoder performs better for this perturbation analysis because:")
        if scores['ae_reconstruction']: print("✓ Superior reconstruction quality")
        if scores['ae_silhouette']: print("✓ Better cluster separation")
        if scores['ae_purity']: print("✓ Better local structure preservation")
        if scores['ae_ground_truth']: print("✓ Higher correlation with true differential expression")
        if scores['ae_stability']: print("✓ More stable across cross-validation")
        if scores['ae_bio_validity']: print("✓ More consistent biological validity")
        
    elif vae_total_score > ae_total_score:
        winner = "VAE"
        print(f"\n🏆 WINNER: {winner}")
        print("\nThe VAE performs better for this perturbation analysis because:")
        if not scores['ae_reconstruction']: print("✓ Superior reconstruction quality")
        if not scores['ae_silhouette']: print("✓ Better cluster separation") 
        if not scores['ae_purity']: print("✓ Better local structure preservation")
        if not scores['ae_ground_truth']: print("✓ Higher correlation with true differential expression")
        if not scores['ae_stability']: print("✓ More stable across cross-validation")
        if not scores['ae_bio_validity']: print("✓ More consistent biological validity")
        print("✓ Can generate new realistic samples (unique advantage)")
        print("✓ Provides uncertainty quantification")
        print("✓ More regularized latent space")
        
    else:
        winner = "TIE"
        print(f"\n🤝 RESULT: {winner}")
        print("\nBoth methods perform comparably. Choice depends on specific needs:")
        print("• Choose AE if: Maximum reconstruction fidelity is critical")
        print("• Choose VAE if: You need generative capabilities or uncertainty quantification")
    
    print(f"\n" + "="*80)
    print("KEY INSIGHTS FOR PERTURBATION BIOLOGY")
    print("="*80)
    
    print(f"\n1. BIOLOGICAL VALIDITY:")
    print(f"   Ground truth correlation - AE: {ae_gt_corr:.3f}, VAE: {vae_gt_corr:.3f}")
    print(f"   This measures how well latent distances reflect true gene expression changes")
    
    print(f"\n2. REPRESENTATION QUALITY:")
    print(f"   Silhouette score - AE: {ae_latent_metrics['silhouette_score']:.3f}, VAE: {vae_latent_metrics['silhouette_score']:.3f}")
    print(f"   Higher scores indicate better separation of perturbation groups")
    
    print(f"\n3. STABILITY:")
    print(f"   CV stability - AE: {cv_results['ae_mean']:.3f}±{cv_results['ae_std']:.3f}")
    print(f"   CV stability - VAE: {cv_results['vae_mean']:.3f}±{cv_results['vae_std']:.3f}")
    print(f"   Lower standard deviation indicates more consistent performance")
    
    print(f"\n4. PERTURBATION DETECTION:")
    ae_detected = sum(ae_results['significant'])
    vae_detected = sum(vae_results['significant'])
    total_perturbations = len(ae_results)
    print(f"   AE detected: {ae_detected}/{total_perturbations} significant perturbations")
    print(f"   VAE detected: {vae_detected}/{total_perturbations} significant perturbations")
    
    print(f"\n5. PRACTICAL CONSIDERATIONS:")
    if vae_generative['mean_correlation'] > 0.8:
        print(f"   ✓ VAE generates realistic samples (correlation: {vae_generative['mean_correlation']:.3f})")
    else:
        print(f"   ⚠ VAE generative quality needs improvement (correlation: {vae_generative['mean_correlation']:.3f})")
    
    print(f"   Reconstruction quality - AE: {ae_recon_mse:.4f}, VAE: {vae_recon_mse:.4f}")
    
    print(f"\n" + "="*80)
    print("METHODOLOGY NOTE")
    print("="*80)
    print("This comparison uses multiple validation approaches:")
    print("• Reconstruction fidelity (how well models preserve original data)")
    print("• Cluster separation (silhouette score for perturbation groups)")
    print("• Local structure preservation (neighborhood purity)")
    print("• Biological ground truth (correlation with differential expression)")
    print("• Cross-validation stability (consistency across data splits)")
    print("• Generative quality (VAE's ability to create realistic samples)")
    print("\nThe 'ground truth correlation' is most important for biological validity,")
    print("as it measures whether latent space distances reflect real gene expression changes.")
    
    return {
        'winner': winner,
        'ae_metrics': ae_metrics,
        'vae_metrics': vae_metrics,
        'cv_results': cv_results,
        'summary_table': summary_table
    }



if __name__ == "__main__":
    main()