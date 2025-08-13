import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from umap import UMAP
import warnings
warnings.filterwarnings('ignore')

# Import all functions from your simulation script
# Make sure the file is in the same directory or adjust the path
try:
    from simulation_processing import (
        simulate_perturbseq_data,
        preprocess_data,
        Autoencoder,
        VAE,
        vae_loss_function,
        train_autoencoder,
        train_vae,
        get_embeddings,
        analyze_perturbation_effects,
        evaluate_reconstruction_quality,
        calculate_latent_space_metrics,
        perform_differential_expression_validation,
        test_generative_capability
    )
    st.success("✅ Successfully imported simulation functions!")
except ImportError as e:
    st.error(f"❌ Could not import simulation_processing.py: {e}")
    st.info("Make sure simulation_processing.py is in the same directory as this streamlit app")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="🧬 VAE Parameter Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧬 VAE Parameter Explorer for Perturb-seq</h1>', unsafe_allow_html=True)

st.markdown("""
**Explore how VAE hyperparameters affect perturbation analysis in single-cell RNA-seq data**

This interactive app lets you adjust key parameters and see their real-time effects on:
- Model training dynamics
- Latent space organization  
- Perturbation detection sensitivity
- Biological validity metrics
""")

# Initialize session state for caching results
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# SIDEBAR PARAMETERS
# =============================================================================

st.sidebar.markdown("## 🎛️ Experiment Parameters")

# Dataset parameters
st.sidebar.markdown("### 📊 Dataset Configuration")
n_cells = st.sidebar.slider(
    "Total number of cells", 
    min_value=500, max_value=3000, value=2000, step=100,
    help="More cells = more statistical power but slower training"
)

n_guides = st.sidebar.slider(
    "Number of sgRNA types", 
    min_value=5, max_value=25, value=15, step=1,
    help="Number of different perturbations to simulate"
)

cells_per_guide = st.sidebar.slider(
    "Average cells per sgRNA", 
    min_value=20, max_value=200, value=80, step=10,
    help="How many cells per perturbation (affects detection power)"
)

# VAE parameters
st.sidebar.markdown("### 🧠 VAE Configuration")
beta = st.sidebar.slider(
    "Beta (KL weight)", 
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Controls reconstruction vs regularization trade-off"
)

latent_dim = st.sidebar.slider(
    "Latent dimensions", 
    min_value=10, max_value=100, value=50, step=10,
    help="Size of the compressed representation"
)

# Training parameters
st.sidebar.markdown("### ⚙️ Training Configuration")
epochs = st.sidebar.slider(
    "Training epochs", 
    min_value=20, max_value=150, value=80, step=10,
    help="More epochs = better training but slower"
)

learning_rate = st.sidebar.selectbox(
    "Learning rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    index=2,
    help="How fast the model learns"
)

# =============================================================================
# HELPER FUNCTIONS FOR STREAMLIT
# =============================================================================

@st.cache_data
def generate_and_process_data(n_cells, n_guides, cells_per_guide):
    """Generate and preprocess data with caching"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Adjust cell distribution
    control_fraction = 0.3
    n_control = int(n_cells * control_fraction)
    n_perturbed = n_cells - n_control
    
    # Generate data
    expression_data, metadata, gene_names = simulate_perturbseq_data(
        n_cells=n_cells, 
        n_genes=4000,  # Reduced for speed
        n_guide_types=n_guides + 1  # +1 for control
    )
    
    # Preprocess
    processed_data, hvg_indices, scaler = preprocess_data(expression_data, n_hvgs=1500)
    
    return expression_data, processed_data, metadata, gene_names, hvg_indices, scaler

def train_models_with_progress(processed_data, beta, latent_dim, epochs, learning_rate):
    """Train models with progress bars"""
    input_dim = processed_data.shape[1]
    
    # Create data loaders
    tensor_data = torch.FloatTensor(processed_data)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Initialize models
    ae_model = Autoencoder(input_dim, latent_dim).to(device)
    vae_model = VAE(input_dim, latent_dim).to(device)
    
    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train AE
    status_text.text("Training Autoencoder...")
    ae_losses = train_autoencoder(ae_model, train_loader, epochs=epochs, lr=learning_rate)
    progress_bar.progress(0.5)
    
    # Train VAE
    status_text.text("Training VAE...")
    vae_losses, vae_recon_losses, vae_kl_losses = train_vae(
        vae_model, train_loader, epochs=epochs, lr=learning_rate, beta=beta
    )
    progress_bar.progress(1.0)
    status_text.text("Training complete!")
    
    return (ae_model, vae_model, eval_loader, 
            ae_losses, vae_losses, vae_recon_losses, vae_kl_losses)

def create_training_plots(ae_losses, vae_losses, vae_recon_losses, vae_kl_losses):
    """Create interactive training plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AE Training Loss', 'VAE Total Loss', 
                       'VAE Loss Components', 'Loss Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    epochs_range = list(range(len(ae_losses)))
    
    # AE loss
    fig.add_trace(
        go.Scatter(x=epochs_range, y=ae_losses, name="AE Loss", line_color='blue'),
        row=1, col=1
    )
    
    # VAE total loss
    fig.add_trace(
        go.Scatter(x=epochs_range, y=vae_losses, name="VAE Total", line_color='red'),
        row=1, col=2
    )
    
    # VAE components
    fig.add_trace(
        go.Scatter(x=epochs_range, y=vae_recon_losses, name="Reconstruction", line_color='green'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs_range, y=vae_kl_losses, name="KL Divergence", line_color='purple'),
        row=2, col=1, secondary_y=True
    )
    
    # Normalized comparison
    ae_norm = np.array(ae_losses) / ae_losses[0] if ae_losses[0] > 0 else ae_losses
    vae_norm = np.array(vae_losses) / vae_losses[0] if vae_losses[0] > 0 else vae_losses
    
    fig.add_trace(
        go.Scatter(x=epochs_range, y=ae_norm, name="AE (normalized)", line_color='blue'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs_range, y=vae_norm, name="VAE (normalized)", line_color='red'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Training Dynamics")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    
    return fig

def create_latent_space_plot(embeddings, metadata, title):
    """Create interactive UMAP plot of latent space"""
    
    # Compute UMAP
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = umap_reducer.fit_transform(embeddings)
    
    # Create plot data
    plot_df = pd.DataFrame({
        'UMAP_1': umap_coords[:, 0],
        'UMAP_2': umap_coords[:, 1],
        'Guide': metadata['guide'].values,
        'Is_Control': metadata['is_control'].values
    })
    
    # Create plotly figure
    fig = px.scatter(
        plot_df, 
        x='UMAP_1', y='UMAP_2', 
        color='Guide',
        title=title,
        hover_data=['Is_Control'],
        width=600, height=500
    )
    
    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Perturbation"
    )
    
    return fig

def create_metrics_comparison(ae_metrics, vae_metrics):
    """Create comparison of model metrics"""
    
    metrics_data = {
        'Metric': ['Reconstruction MSE', 'Silhouette Score', 'Neighborhood Purity', 'Ground Truth Correlation'],
        'Autoencoder': [
            ae_metrics['reconstruction_mse'],
            ae_metrics['latent_metrics']['silhouette_score'],
            ae_metrics['latent_metrics']['neighborhood_purity'],
            ae_metrics['ground_truth_correlation']
        ],
        'VAE': [
            vae_metrics['reconstruction_mse'],
            vae_metrics['latent_metrics']['silhouette_score'],
            vae_metrics['latent_metrics']['neighborhood_purity'],
            vae_metrics['ground_truth_correlation']
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Determine winners
    df['Winner'] = ['AE' if ae < vae else 'VAE' 
                    if metric == 'Reconstruction MSE' 
                    else 'VAE' if vae > ae else 'AE' 
                    for ae, vae, metric in zip(df['Autoencoder'], df['VAE'], df['Metric'])]
    
    return df

# =============================================================================
# MAIN APP LOGIC
# =============================================================================

# Create parameter hash for caching
current_params = (n_cells, n_guides, cells_per_guide, beta, latent_dim, epochs, learning_rate)
params_changed = st.session_state.last_params != current_params

if st.sidebar.button("🚀 Run Analysis", type="primary") or params_changed:
    
    st.session_state.last_params = current_params
    
    with st.spinner("🔬 Generating synthetic Perturb-seq data..."):
        expression_data, processed_data, metadata, gene_names, hvg_indices, scaler = generate_and_process_data(
            n_cells, n_guides, cells_per_guide
        )
    
    # Display dataset info
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cells", n_cells)
    with col2:
        st.metric("sgRNA Types", len(metadata['guide'].unique()))
    with col3:
        st.metric("Genes (HVG)", processed_data.shape[1])
    with col4:
        control_count = sum(metadata['is_control'])
        st.metric("Control Cells", f"{control_count} ({control_count/n_cells*100:.1f}%)")
    
    # Train models
    st.markdown('<div class="section-header">🧠 Model Training</div>', unsafe_allow_html=True)
    
    with st.spinner("Training models... This may take a minute."):
        (ae_model, vae_model, eval_loader, 
         ae_losses, vae_losses, vae_recon_losses, vae_kl_losses) = train_models_with_progress(
            processed_data, beta, latent_dim, epochs, learning_rate
        )
    
    # Show training plots
    training_fig = create_training_plots(ae_losses, vae_losses, vae_recon_losses, vae_kl_losses)
    st.plotly_chart(training_fig, use_container_width=True)
    
    # Extract embeddings and run analysis
    with st.spinner("🔍 Analyzing latent representations..."):
        ae_embeddings = get_embeddings(ae_model, eval_loader, 'AE')
        vae_embeddings = get_embeddings(vae_model, eval_loader, 'VAE')
        
        # Calculate metrics
        ae_recon_mse = evaluate_reconstruction_quality(ae_model, eval_loader, 'AE')
        vae_recon_mse = evaluate_reconstruction_quality(vae_model, eval_loader, 'VAE')
        
        ae_latent_metrics = calculate_latent_space_metrics(ae_embeddings, metadata)
        vae_latent_metrics = calculate_latent_space_metrics(vae_embeddings, metadata)
        
        ae_gt_corr, _, _ = perform_differential_expression_validation(
            processed_data, ae_embeddings, metadata, expression_data, hvg_indices
        )
        vae_gt_corr, _, _ = perform_differential_expression_validation(
            processed_data, vae_embeddings, metadata, expression_data, hvg_indices
        )
        
        # Compile metrics
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
        
        # Perturbation analysis
        ae_results, _ = analyze_perturbation_effects(ae_embeddings, metadata)
        vae_results, _ = analyze_perturbation_effects(vae_embeddings, metadata)
    
    # =============================================================================
    # RESULTS VISUALIZATION
    # =============================================================================
    
    st.markdown('<div class="section-header">📈 Results Comparison</div>', unsafe_allow_html=True)
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance Metrics")
        metrics_df = create_metrics_comparison(ae_metrics, vae_metrics)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Winner summary
        ae_wins = sum(metrics_df['Winner'] == 'AE')
        vae_wins = sum(metrics_df['Winner'] == 'VAE')
        
        if ae_wins > vae_wins:
            st.success(f"🏆 **Autoencoder wins** ({ae_wins}/{len(metrics_df)} metrics)")
        elif vae_wins > ae_wins:
            st.success(f"🏆 **VAE wins** ({vae_wins}/{len(metrics_df)} metrics)")
        else:
            st.info(f"🤝 **Tie** ({ae_wins}-{vae_wins})")
    
    with col2:
        st.subheader("🔍 Perturbation Detection")
        
        ae_detected = sum(ae_results['significant'])
        vae_detected = sum(vae_results['significant'])
        total_perturbations = len(ae_results)
        
        detection_data = {
            'Method': ['Autoencoder', 'VAE'],
            'Significant_Perturbations': [ae_detected, vae_detected],
            'Detection_Rate': [ae_detected/total_perturbations*100, vae_detected/total_perturbations*100]
        }
        
        detection_df = pd.DataFrame(detection_data)
        st.dataframe(detection_df, use_container_width=True)
        
        # Detection comparison chart
        fig_detection = px.bar(
            detection_df, 
            x='Method', y='Significant_Perturbations',
            title='Significant Perturbations Detected',
            color='Method',
            color_discrete_map={'Autoencoder': 'blue', 'VAE': 'red'}
        )
        st.plotly_chart(fig_detection, use_container_width=True)
    
    # Latent space visualizations
    st.markdown('<div class="section-header">🗺️ Latent Space Organization</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ae_plot = create_latent_space_plot(ae_embeddings, metadata, "Autoencoder Latent Space")
        st.plotly_chart(ae_plot, use_container_width=True)
    
    with col2:
        vae_plot = create_latent_space_plot(vae_embeddings, metadata, f"VAE Latent Space (β={beta})")
        st.plotly_chart(vae_plot, use_container_width=True)
    
    # Beta effect analysis
    st.markdown('<div class="section-header">🎛️ Beta Parameter Impact</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Final Reconstruction Loss", 
            f"{vae_recon_losses[-1]:.2f}",
            help="Lower is better (how well VAE recreates input)"
        )
    
    with col2:
        st.metric(
            "Final KL Divergence", 
            f"{vae_kl_losses[-1]:.2f}",
            help="Measures latent space regularization"
        )
    
    with col3:
        ratio = vae_kl_losses[-1] / (vae_recon_losses[-1] + 1e-8)
        st.metric(
            "KL/Reconstruction Ratio", 
            f"{ratio:.3f}",
            help="Balance between regularization and fidelity"
        )
    
    # Beta interpretation
    if beta < 0.5:
        st.info("🔍 **Low Beta**: Prioritizes reconstruction quality, may sacrifice latent structure")
    elif beta > 2.0:
        st.warning("⚠️ **High Beta**: Strong regularization, may lose important biological information")
    else:
        st.success("✅ **Balanced Beta**: Good trade-off between reconstruction and structure")
    
    # Save results to session state
    st.session_state.results_cache[current_params] = {
        'ae_metrics': ae_metrics,
        'vae_metrics': vae_metrics,
        'ae_results': ae_results,
        'vae_results': vae_results
    }

# =============================================================================
# SIDEBAR INTERPRETATION GUIDE
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("## 💡 Parameter Guide")

with st.sidebar.expander("🎯 Beta Parameter"):
    st.markdown("""
    **β < 1**: Under-regularized
    - Better reconstruction
    - Less structured latent space
    - May overfit to noise
    
    **β = 1**: Standard VAE
    - Balanced trade-off
    - Good starting point
    
    **β > 1**: Over-regularized  
    - More structured latent space
    - May lose biological signal
    - Better for generation
    """)

with st.sidebar.expander("📊 Cell Number Effects"):
    st.markdown("""
    **More cells per sgRNA**:
    - Higher statistical power
    - Better perturbation detection
    - More stable results
    
    **Fewer cells per sgRNA**:
    - Faster training
    - More realistic for rare perturbations
    - Higher variance in results
    """)

with st.sidebar.expander("🧠 Latent Dimensions"):
    st.markdown("""
    **Higher dimensions**:
    - Can capture more biological complexity
    - Risk of overfitting
    - Harder to interpret
    
    **Lower dimensions**:
    - More compressed representation
    - Better generalization
    - Easier to visualize
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🧬 Interactive VAE Explorer for Perturbation Biology<br>
    Adjust parameters in the sidebar and click "Run Analysis" to explore!
</div>
""", unsafe_allow_html=True)

