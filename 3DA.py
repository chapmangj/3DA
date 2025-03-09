import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from streamlit import session_state as state
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from PIL import Image


# Init session state vars
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'wcss' not in st.session_state:
    st.session_state.wcss = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3
if 'selected_cluster_features' not in st.session_state:
    st.session_state.selected_cluster_features = None
if 'apply_filters_globally' not in st.session_state:
    st.session_state.apply_filters_globally = False
if 'previous_collar_file' not in st.session_state:
    st.session_state.previous_collar_file = None
if 'previous_assay_file' not in st.session_state:
    st.session_state.previous_assay_file = None
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'viz_df' not in st.session_state:
    st.session_state.viz_df = None
if 'viz_litho_df' not in st.session_state:
    st.session_state.viz_litho_df = None
if 'collar_df' not in st.session_state:
    st.session_state.collar_df = None
if 'element_cols' not in st.session_state:
    st.session_state.element_cols = []
if 'litho_dict' not in st.session_state:
    st.session_state.litho_dict = None
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = None
if 'significant_intervals' not in st.session_state:
    st.session_state.significant_intervals = None

# Setup page
st.set_page_config(
    page_title="3DA",
    page_icon="🔍",
    layout="wide"
)
st.markdown('<div style="position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;">Version 0.8</div>', unsafe_allow_html=True)


# Centre logo and header
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    try:
        logo = Image.open("3DA logo.png")
        st.image(logo, use_container_width=True)
    except:
        st.write("3DA")

st.markdown("<h1 style='text-align: center;'>Exploratory Data Analysis and Visualisation</h1>", unsafe_allow_html=True)

# 1. DATA READING & PROCESSING FUNCTIONS
def read_csv(file, format_type):
    """Read CSV files with standard or mining format"""
    try:
        if format_type == "Standard CSV (Headers in row 1)":
            return pd.read_csv(file)
        else:  # Geological Survey Format
            df = pd.read_csv(file, header=None)
            header_row_idx = df[df.iloc[:, 0] == 'H1000'].index[0]
            headers = df.iloc[header_row_idx].values.tolist()
            data_rows = df[df.iloc[:, 0] == 'D']
            result_df = pd.DataFrame(data_rows.values, columns=headers)
            return result_df.iloc[:, 1:]
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def process_collar_data(collar_file, format_type):
    """Process collar file and return formatted DataFrame."""
    try:
        collar_df = read_csv(collar_file, format_type)
        if collar_df is not None:
            st.write("Collar Data Preview:")
            st.write(collar_df.head())

            st.subheader("Select Collar Columns")
            # Try to find default columns
            hole_id_col = next((col for col in collar_df.columns if 'hole' in col.lower()), None)
            easting_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['east', 'mga_e', 'x'])), None)
            northing_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['north', 'mga_n', 'y'])), None)
            elevation_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['elevation', 'rl', 'z'])), None)
            dip_col = next((col for col in collar_df.columns if 'dip' in col.lower()), None)
            azimuth_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['azi', 'azimuth'])), None)

            hole_id_col = st.selectbox("Select HOLE_ID column", collar_df.columns, 
                                       index=(collar_df.columns.get_loc(hole_id_col) if hole_id_col else 0))
            easting_col = st.selectbox("Select EASTING column", collar_df.columns,
                                       index=(collar_df.columns.get_loc(easting_col) if easting_col else 0))
            northing_col = st.selectbox("Select NORTHING column", collar_df.columns,
                                        index=(collar_df.columns.get_loc(northing_col) if northing_col else 0))
            elevation_col = st.selectbox("Select ELEVATION column", collar_df.columns,
                                         index=(collar_df.columns.get_loc(elevation_col) if elevation_col else 0))
            dip_col = st.selectbox("Select DIP column", collar_df.columns,
                                   index=(collar_df.columns.get_loc(dip_col) if dip_col else 0))
            azimuth_col = st.selectbox("Select AZIMUTH column", collar_df.columns,
                                       index=(collar_df.columns.get_loc(azimuth_col) if azimuth_col else 0))

            collar_df = collar_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                easting_col: 'EASTING',
                northing_col: 'NORTHING',
                elevation_col: 'ELEVATION',
                dip_col: 'DIP',
                azimuth_col: 'AZIMUTH'
            })
            for col in ['EASTING', 'NORTHING', 'ELEVATION', 'DIP', 'AZIMUTH']:
                collar_df[col] = pd.to_numeric(collar_df[col], errors='coerce')
            
            return collar_df
    except Exception as e:
        st.error(f"Error processing collar file: {str(e)}")
        return None

def process_assay_data(assay_file, format_type):
    """Process assay file and select columns for geochemical data."""
    try:
        assay_df = read_csv(assay_file, format_type)
        if assay_df is not None:
            st.write("Assay Data Preview:")
            st.write(assay_df.head())

            st.subheader("Select Assay Columns")
            hole_id_col = next((col for col in assay_df.columns if 'hole' in col.lower()), None)
            from_col = next((col for col in assay_df.columns if 'from' in col.lower()), None)
            to_col = next((col for col in assay_df.columns if 'to' in col.lower()), None)

            hole_id_col = st.selectbox("Select HOLE_ID column (Assay)", assay_df.columns, 
                                       index=(assay_df.columns.get_loc(hole_id_col) if hole_id_col else 0))
            from_col = st.selectbox("Select FROM column", assay_df.columns, 
                                    index=(assay_df.columns.get_loc(from_col) if from_col else 0))
            to_col = st.selectbox("Select TO column", assay_df.columns, 
                                  index=(assay_df.columns.get_loc(to_col) if to_col else 0))

            available_elements = [col for col in assay_df.columns if col not in [hole_id_col, from_col, to_col]]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                select_all = st.checkbox("Select all (Remember to remove non-assay columns)", value=True)
            with col2:
                if select_all:
                    element_cols = st.multiselect(
                        "Select element columns",
                        available_elements,
                        default=available_elements
                    )
                else:
                    element_cols = st.multiselect(
                        "Select element columns",
                        available_elements
                    )

            assay_df = assay_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                from_col: 'FROM',
                to_col: 'TO'
            })
            
            # Handle below detection limits
            numeric_cols = ['FROM', 'TO'] + element_cols
            for col in numeric_cols:
                # Replace "<" with "-" to detect below detection
                assay_df[col] = assay_df[col].astype(str).str.replace('<', '-')
                # Convert negative to half detection limit
                assay_df[col] = pd.to_numeric(assay_df[col], errors='coerce')
                assay_df.loc[assay_df[col] < 0, col] = abs(assay_df[col]) / 2

            assay_df = assay_df[['HOLE_ID', 'FROM', 'TO'] + element_cols]
            return assay_df, element_cols
    except Exception as e:
        st.error(f"Error processing assay file: {str(e)}")
        return None, None

def process_litho_data(litho_file, format_type):
    """Process lithology file and return formatted DataFrame."""
    try:
        litho_df = read_csv(litho_file, format_type)
        if litho_df is not None:
            st.write("Lithology Data Preview:")
            st.write(litho_df.head())

            st.subheader("Select Lithology Columns")
            hole_id_col = next((col for col in litho_df.columns if 'hole' in col.lower()), None)
            from_col = next((col for col in litho_df.columns if 'from' in col.lower()), None)
            to_col = next((col for col in litho_df.columns if 'to' in col.lower()), None)
            litho_col = next((col for col in litho_df.columns if any(x in col.lower() for x in ['lith', 'rock', 'geol'])), None)

            hole_id_col = st.selectbox("Select HOLE_ID column (Lithology)", litho_df.columns, 
                                       index=(litho_df.columns.get_loc(hole_id_col) if hole_id_col else 0))
            from_col = st.selectbox("Select FROM column (Lithology)", litho_df.columns, 
                                    index=(litho_df.columns.get_loc(from_col) if from_col else 0))
            to_col = st.selectbox("Select TO column (Lithology)", litho_df.columns, 
                                  index=(litho_df.columns.get_loc(to_col) if to_col else 0))
            litho_col = st.selectbox("Select LITHOLOGY column", litho_df.columns, 
                                     index=(litho_df.columns.get_loc(litho_col) if litho_col else 0))

            litho_df = litho_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                from_col: 'FROM',
                to_col: 'TO',
                litho_col: 'LITHO'
            })
            litho_df['FROM'] = pd.to_numeric(litho_df['FROM'], errors='coerce')
            litho_df['TO'] = pd.to_numeric(litho_df['TO'], errors='coerce')
            litho_df = litho_df[['HOLE_ID', 'FROM', 'TO', 'LITHO']]
            return litho_df
    except Exception as e:
        st.error(f"Error processing lithology file: {str(e)}")
        return None

def process_litho_dict(litho_dict_file, format_type):
    """Process lithology dictionary file and return a dictionary code->description."""
    try:
        litho_dict_df = read_csv(litho_dict_file, format_type)
        if litho_dict_df is not None:
            st.write("Lithology Dictionary Preview:")
            st.write(litho_dict_df.head())

            st.subheader("Select Lithology Dictionary Columns")
            code_col = next((col for col in litho_dict_df.columns if any(x in col.lower() for x in ['code', 'lith', 'rock'])), None)
            desc_col = next((col for col in litho_dict_df.columns if any(x in col.lower() for x in ['desc', 'name', 'type'])), None)

            code_col = st.selectbox("Select Lithology Code column", litho_dict_df.columns,
                                    index=(litho_dict_df.columns.get_loc(code_col) if code_col else 0))
            desc_col = st.selectbox("Select Lithology Description column", litho_dict_df.columns,
                                    index=(litho_dict_df.columns.get_loc(desc_col) if desc_col else 0))

            litho_dict = dict(zip(litho_dict_df[code_col], litho_dict_df[desc_col]))
            return litho_dict
    except Exception as e:
        st.error(f"Error processing lithology dictionary file: {str(e)}")
        return None

# 2. COMPOSITING ROUTINE
def composite_geochemical_data(df, element_cols, composite_length):
    """Create composites of geochemical intervals at a fixed length"""
    if 'HOLE_ID' not in df.columns or 'FROM' not in df.columns or 'TO' not in df.columns:
        st.error("DataFrame must have columns 'HOLE_ID', 'FROM', 'TO' for compositing.")
        return df

    composited_rows = []

    # Process each hole individually
    for hole_id, hole_data in df.groupby('HOLE_ID', sort=False):
        hole_data = hole_data.sort_values('FROM')
        hole_start = hole_data['FROM'].min()
        hole_end = hole_data['TO'].max()
        composite_top = hole_start

        while composite_top < hole_end:
            composite_bot = composite_top + composite_length
            
            if composite_bot > hole_end:
                composite_bot = hole_end

            overlap = hole_data[
                (hole_data['FROM'] < composite_bot) &
                (hole_data['TO'] > composite_top)
            ].copy()

            if overlap.empty:
                composite_top = composite_bot
                continue

            overlap['interval_start'] = overlap['FROM'].clip(lower=composite_top)
            overlap['interval_end'] = overlap['TO'].clip(upper=composite_bot)
            overlap['interval_length'] = overlap['interval_end'] - overlap['interval_start']

            composited_values = {}
            total_length = overlap['interval_length'].sum()

            for elem in element_cols:
                composited_values[elem] = np.average(overlap[elem], weights=overlap['interval_length'])

            composited_row = {
                'HOLE_ID': hole_id,
                'FROM': composite_top,
                'TO': composite_bot
            }
            composited_row.update(composited_values)
            composited_rows.append(composited_row)
            composite_top = composite_bot

    composite_df = pd.DataFrame(composited_rows)
    return composite_df

# 3. PLOTTING & UTILITY FUNCTIONS
def create_swath_plots(merged_df, primary_element, use_log_scale):
    """Create swath plots for Easting, Northing, and Elevation"""
    st.header("Swath Plots")

    st.subheader("Swath Plot Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        easting_bins = st.number_input("Number of Easting bins", min_value=2, max_value=50, value=5)
    with col2:
        northing_bins = st.number_input("Number of Northing bins", min_value=2, max_value=50, value=5)
    with col3:
        elevation_bins = st.number_input("Number of Elevation bins", min_value=2, max_value=50, value=5)

    tab1, tab2, tab3 = st.tabs(["Easting Swath", "Northing Swath", "Elevation Swath"])

    def create_swath_data(df, coord_col, value_col, num_bins=3):
        df = df.sort_values(coord_col)
        bins = np.linspace(df[coord_col].min(), df[coord_col].max(), num_bins + 1)
        df['bin'] = pd.cut(df[coord_col], bins)
        swath_stats = df.groupby('bin')[value_col].agg(['mean', 'count', 'std']).reset_index()
        swath_stats['bin_center'] = [(x.left + x.right)/2 for x in swath_stats['bin']]
        swath_stats['bin_width'] = [(x.right - x.left) for x in swath_stats['bin']]
        swath_stats = swath_stats[swath_stats['count'] > 0]
        return swath_stats

    def plot_swath(swath_stats, title, x_title, y_title, reverse_x=False):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=swath_stats['bin_center'],
            y=swath_stats['count'],
            name='Number of Samples',
            yaxis='y2',
            opacity=0.3,
            marker_color='lightblue',
            width=swath_stats['bin_width'] * 0.9  # Make bars wider (90% of bin width)
        ))
        fig.add_trace(go.Scatter(
            x=swath_stats['bin_center'],
            y=swath_stats['mean'],
            mode='markers+lines',
            name='Mean Grade',
            line=dict(color='blue', width=2),
            error_y=dict(
                type='data',
                array=swath_stats['std'],
                visible=True,
                color='red',
                thickness=1,
                width=4
            )
        ))
        layout_dict = {
            'xaxis_title': x_title,
            'yaxis_title': y_title,
            'yaxis2': dict(
                title="Number of Samples",
                overlaying='y',
                side='right'
            ),
            'height': 700,  
            'width': 2000,  
            'title': title
        }
        if reverse_x:
            layout_dict['xaxis'] = {'autorange': 'reversed'}
        fig.update_layout(**layout_dict)
        return fig

    with tab1:
        easting_swath = create_swath_data(merged_df, 'x', primary_element, easting_bins)
        fig = plot_swath(easting_swath, "Grade Distribution by Easting", "Easting", primary_element)
        st.plotly_chart(fig)

    with tab2:
        northing_swath = create_swath_data(merged_df, 'y', primary_element, northing_bins)
        fig = plot_swath(northing_swath, "Grade Distribution by Northing", "Northing", primary_element)
        st.plotly_chart(fig)

    with tab3:
        elevation_swath = create_swath_data(merged_df, 'z', primary_element, elevation_bins)
        fig = plot_swath(elevation_swath, "Grade Distribution by Elevation", "Elevation", primary_element, reverse_x=True)
        st.plotly_chart(fig)

def perform_clustering(df, features, max_clusters=10, use_log_transform=False):
    """Run k-means clustering analysis"""
    X = df[features]
    if use_log_transform:
        X = np.log(X + 1e-10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return X_scaled, scaler, wcss

def plot_scree(wcss=None, explained_variance_ratio=None, is_pca=False):
    """Create scree plot for clustering or PCA"""
    fig = go.Figure()
    if is_pca and explained_variance_ratio is not None:
        # PCA scree plot
        cumulative = np.cumsum(explained_variance_ratio)
        fig.add_trace(go.Bar(
            x=list(range(1, len(explained_variance_ratio) + 1)),
            y=explained_variance_ratio,
            name='Individual'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative,
            name='Cumulative',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='PCA Scree Plot',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            showlegend=True
        )
    elif wcss is not None:
        # Clustering scree plot
        wcss_normalised = [w / wcss[0] for w in wcss]
        wcss_decrease = [-((wcss[i] - wcss[i-1]) / wcss[i-1]) * 100 if i > 0 else 0 
                         for i in range(len(wcss))]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(wcss) + 1)),
                y=wcss_normalised,
                mode='lines+markers',
                name='Normalised WCSS',
                line=dict(color='blue')
            ), secondary_y=False
        )
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(wcss) + 1)),
                y=wcss_decrease,
                name='% Decrease',
                marker_color='rgba(255, 165, 0, 0.5)'
            ), secondary_y=True
        )
        fig.update_layout(
            title='Clustering Scree Plot',
            xaxis_title='Number of Clusters',
            showlegend=True
        )
        fig.update_yaxes(title_text="Normalised WCSS", secondary_y=False)
        fig.update_yaxes(title_text="Percentage Decrease", secondary_y=True)
    return fig

def perform_pca_analysis(X, n_components, use_log_transform=False):
    """Run PCA with optional log transform"""
    if use_log_transform:
        X = np.log(X + 1e-10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler

def plot_pca_biplot(X_pca, pca, feature_names):
    """Create PCA biplot (2D or 3D)"""
    fig = go.Figure()
    n_components = X_pca.shape[1]
    if n_components >= 3:
        fig.add_trace(go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(size=6),
            name='Samples',
            opacity=0.7
        ))
    else:
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(size=6),
            name='Samples',
            opacity=0.7
        ))

    data_range = np.max(np.abs(X_pca))
    loading_range = np.max(np.abs(pca.components_))
    scaling_factor = (data_range / loading_range) * 0.8

    for i, feature in enumerate(feature_names):
        if n_components >= 3:
            fig.add_trace(go.Scatter3d(
                x=[0, pca.components_[0, i] * scaling_factor],
                y=[0, pca.components_[1, i] * scaling_factor],
                z=[0, pca.components_[2, i] * scaling_factor],
                mode='lines+text',
                line=dict(color='red', width=3),
                text=['', feature],
                textposition='top center',
                textfont=dict(size=12),
                name=feature,
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[0, pca.components_[0, i] * scaling_factor],
                y=[0, pca.components_[1, i] * scaling_factor],
                mode='lines+text',
                line=dict(color='red', width=3),
                text=['', feature],
                textposition='top center',
                textfont=dict(size=12),
                name=feature,
                showlegend=True
            ))

    if n_components >= 3:
        fig.update_layout(
            title='3D PCA Biplot',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1600,
            height=1200
        )
    else:
        fig.update_layout(
            title='2D PCA Biplot',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
            width=1200,
            height=800
        )
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=12)
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.add_annotation(
        text="Red lines show feature loadings",
        xref="paper", yref="paper",
        x=0, y=1.05,
        showarrow=False,
        font=dict(size=14)
    )
    return fig

def plot_eigenvectors(pca, feature_names):
    """Plot eigenvectors for each principal component"""
    n_components = pca.components_.shape[0]
    n_rows = (n_components + 1) // 2
    fig = make_subplots(rows=n_rows, cols=2)
    for i in range(n_components):
        row = i // 2 + 1
        col = i % 2 + 1
        eigenvector = pca.components_[i]
        sorted_idx = np.argsort(eigenvector)
        pos = np.arange(len(eigenvector))
        fig.add_trace(
            go.Scatter(
                x=pos,
                y=eigenvector[sorted_idx],
                mode='lines+markers',
                name=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        fig.update_xaxes(
            ticktext=[feature_names[idx] for idx in sorted_idx],
            tickvals=pos,
            tickangle=45,
            row=row, col=col
        )
        fig.update_yaxes(title_text=f'PC{i+1} Loading', row=row, col=col)
    fig.update_layout(
        height=300 * n_rows,
        width=1600,
        showlegend=True,
        title_text="Principal Component Loadings"
    )
    return fig

def get_cluster_summary(df, cluster_features, primary_element):
    # Get stats for each feature by cluster
    features = list(dict.fromkeys(cluster_features + [primary_element]))
    summary_dict = {}
    for feature in features:
        cluster_stats = {}
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster][feature]
            cluster_stats[f'Cluster {cluster}'] = {
                'mean': cluster_data.mean(),
                'median': cluster_data.median(),
                'std': cluster_data.std(),
                'min': cluster_data.min(),
                'max': cluster_data.max()
            }
        summary_dict[feature] = cluster_stats
    index = pd.MultiIndex.from_product([features, ['mean', 'median', 'std', 'min', 'max']])
    columns = [f'Cluster {i}' for i in sorted(df['Cluster'].unique())]
    summary_df = pd.DataFrame(index=index, columns=columns)
    for feature in features:
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            for cluster in sorted(df['Cluster'].unique()):
                summary_df.loc[(feature, stat), f'Cluster {cluster}'] = summary_dict[feature][f'Cluster {cluster}'][stat]
    return summary_df

def plot_cluster_boxplots(df, cluster_features, primary_element, use_log_scale=True):
    """Create boxplots showing element distributions by cluster"""
    fig = go.Figure()
    cluster_colors = px.colors.qualitative.Set1
    legend_added = set()
    for feature in cluster_features + [primary_element]:
        for i, cluster in enumerate(sorted(df['Cluster'].unique())):
            cluster_data = df[df['Cluster'] == cluster]
            show_legend = cluster not in legend_added
            if show_legend:
                legend_added.add(cluster)
            fig.add_trace(go.Box(
                x=[feature] * len(cluster_data),
                y=cluster_data[feature],
                name=f'Cluster {cluster}',
                marker_color=cluster_colors[i % len(cluster_colors)],
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0,
                offsetgroup=str(cluster),
                showlegend=show_legend
            ))
    fig.update_layout(
        xaxis_title='Elements',
        yaxis_title='Values',
        boxmode='group',
        height=1000,
        width=max(1800, 200 * len(cluster_features)),
        showlegend=True,
        yaxis=dict(
            type='log' if use_log_scale else 'linear',
            tickformat='.3f',
            dtick='D1' if use_log_scale else None,
            showgrid=True,
            tickmode='auto',
            tick0=0,
            ticks='outside'
        ),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        margin=dict(r=150)
    )
    if use_log_scale:
        all_values = []
        for feature in cluster_features + [primary_element]:
            all_values.extend(df[feature].dropna().values)
        min_val = min(v for v in all_values if v > 0)
        max_val = max(all_values)
        log_min = np.floor(np.log10(min_val))
        log_max = np.ceil(np.log10(max_val))
        tick_vals = []
        tick_text = []
        for i in range(int(log_min), int(log_max) + 1):
            tick_vals.extend([10**i, 2*10**i, 5*10**i])
            tick_text.extend([f'{10**i:.3g}', f'{2*10**i:.3g}', f'{5*10**i:.3g}'])
        tick_vals = [v for v in tick_vals if min_val <= v <= max_val]
        tick_text = tick_text[:len(tick_vals)]
        fig.update_yaxes(
            tickvals=tick_vals,
            ticktext=tick_text
        )
    return fig

def plot_lithology_cluster_comparison(df):
    """Create heatmap comparing lithology vs cluster"""
    lith_cluster_counts = df.groupby(['LITHO', 'Cluster']).size().unstack(fill_value=0)
    fig = px.imshow(lith_cluster_counts,
                    labels=dict(x="Cluster", y="Lithology", color="Count"),
                    title="Lithology vs Cluster Heatmap")
    fig.update_traces(
        text=lith_cluster_counts.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 12},
        showscale=True
    )
    fig.update_xaxes(
        tickmode='array',
        ticktext=list(range(len(lith_cluster_counts.columns))),
        tickvals=list(range(len(lith_cluster_counts.columns)))
    )
    return fig

def create_lithology_analysis(merged_df, primary_element, use_log_scale, litho_dict=None):
    """Create lithology analysis plots and stats"""
    st.header("Lithology Analysis")
    st.subheader(f"Summary Statistics by Lithology - {primary_element}")
    
    litho_stats = merged_df.groupby('LITHO').agg({
        primary_element: [
            'count',
            'mean',
            'median',
            'std',
            'min',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            'max',
            lambda x: x.std()/x.mean() if x.mean() != 0 else np.nan
        ]
    })
    litho_stats.columns = [
        'Count', 'Mean', 'Median', 'Std Dev',
        'Min', 'Q1', 'Q3', 'Max', 'CV'
    ]
    litho_stats = litho_stats.round(3)
    litho_stats = litho_stats.reset_index()

    if litho_dict:
        litho_stats['Description'] = litho_stats['LITHO'].map(lambda x: litho_dict.get(x, ""))
        litho_stats['LITHO_LABEL'] = litho_stats.apply(
            lambda row: f"{row['LITHO']} {row['Description']}".strip(), axis=1
        )
    else:
        litho_stats['LITHO_LABEL'] = litho_stats['LITHO']

    litho_stats = litho_stats.sort_values('Count', ascending=False)
    display_columns = ['LITHO']
    if litho_dict:
        display_columns.append('Description')
    display_columns.extend(['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Q1', 'Q3', 'Max', 'CV'])
    st.dataframe(litho_stats[display_columns])

    st.subheader(f"Grade Distribution by Lithology - {primary_element}")
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "Sort Lithologies by:",
            ['Median', 'Mean', 'Count', 'Alphabetical']
        )
    with col2:
        min_samples = st.number_input(
            "Minimum samples per lithology:",
            min_value=1,
            value=2,
            step=1
        )
    valid_lithos = litho_stats[litho_stats['Count'] >= min_samples]['LITHO']
    plot_df = merged_df[merged_df['LITHO'].isin(valid_lithos)].copy()
    if not plot_df.empty:
        if sort_by == 'Median':
            litho_order = litho_stats[litho_stats['Count'] >= min_samples].sort_values('Median', ascending=False)['LITHO']
        elif sort_by == 'Mean':
            litho_order = litho_stats[litho_stats['Count'] >= min_samples].sort_values('Mean', ascending=False)['LITHO']
        elif sort_by == 'Count':
            litho_order = litho_stats[litho_stats['Count'] >= min_samples].sort_values('Count', ascending=False)['LITHO']
        else: 
            litho_order = sorted(valid_lithos)

        fig = go.Figure()
        litho_label_map = dict(zip(litho_stats['LITHO'], litho_stats['LITHO_LABEL']))
        for litho in litho_order:
            litho_data = plot_df[plot_df['LITHO'] == litho][primary_element]
            fig.add_trace(go.Box(
                y=litho_data,
                name=litho_label_map[litho],
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0
            ))
        y_min = plot_df[primary_element].min()
        y_max = plot_df[primary_element].max()
        if use_log_scale and y_min > 0:
            log_y_min = np.floor(np.log10(y_min))
            log_y_max = np.ceil(np.log10(y_max))
            tick_values = [y_min]
            tick_texts = [f'{y_min:.2f}']
            for i in range(int(log_y_min), int(log_y_max) + 1):
                current = 10**i
                if y_min <= current <= y_max:
                    if current not in tick_values:
                        tick_values.append(current)
                        tick_texts.append(f'{current:.0f}')
                if 10 * current <= y_max and 10 * current >= y_min:
                    if 10 * current not in tick_values:
                        tick_values.append(10 * current)
                        tick_texts.append(f'{10 * current:.0f}')
            if y_max not in tick_values:
                tick_values.append(y_max)
                tick_texts.append(f'{y_max:.2f}')
            tick_values, tick_texts = zip(*sorted(zip(tick_values, tick_texts)))
            fig.update_layout(
                title=f"{primary_element} Distribution by Lithology",
                yaxis=dict(
                    title=f"{primary_element} (log scale)",
                    type='log',
                    range=[np.log10(y_min), np.log10(y_max)],
                    showgrid=True,
                    tickmode='array',
                    tickvals=tick_values,
                    ticktext=tick_texts
                ),
                showlegend=True,
                height=600,
                boxmode='group'
            )
        else:
            y_range = y_max - y_min
            if y_range <= 10:
                tick_interval = 1
            elif y_range <= 20:
                tick_interval = 2
            elif y_range <= 50:
                tick_interval = 5
            else:
                tick_interval = 10 ** np.floor(np.log10(y_range / 10))
            tick_min = np.floor(y_min / tick_interval) * tick_interval
            tick_max = np.ceil(y_max / tick_interval) * tick_interval
            tick_values = np.arange(tick_min, tick_max + tick_interval, tick_interval)
            if y_min not in tick_values:
                tick_values = np.sort(np.append(tick_values, y_min))
            if y_max not in tick_values:
                tick_values = np.sort(np.append(tick_values, y_max))
            if tick_interval >= 1:
                tick_format = '.0f'
            elif tick_interval >= 0.1:
                tick_format = '.1f'
            else:
                tick_format = '.2f'
            tick_texts = [f'{v:.2f}' if v in (y_min, y_max) else f'{v:{tick_format}}' for v in tick_values]
            fig.update_layout(
                title=f"{primary_element} Distribution by Lithology",
                yaxis=dict(
                    title=primary_element,
                    type='linear',
                    range=[tick_values[0], tick_values[-1]],
                    showgrid=True,
                    tickmode='array',
                    tickvals=tick_values,
                    ticktext=tick_texts
                ),
                showlegend=True,
                height=600,
                boxmode='group'
            )
        fig.update_layout(
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10)
            )
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for plotting after applying filters.")

def add_grade_visualisation(fig, viz_df, primary_element, use_log_scale, viz_mode, color_by='grade', x_offset=0):
    """Add grade or cluster visualisation to the figure"""
    valid_samples = viz_df
    if color_by == 'cluster':
        color_values = valid_samples['Cluster']
        color_bar_title = 'Cluster'
        custom_colorscale = px.colors.qualitative.G10
    else:
        if use_log_scale:
            valid_samples = valid_samples[valid_samples[primary_element] > 0]
            color_values = valid_samples[primary_element]
            color_bar_title = primary_element
            min_val = valid_samples[primary_element].min()
            max_val = valid_samples[primary_element].max()
            log_min = np.floor(np.log10(min_val))
            log_max = np.ceil(np.log10(max_val))
            tick_vals = []
            tick_text = []
            def format_number(x):
                if x >= 1:
                    return f'{x:.0f}'
                elif x >= 0.1:
                    return f'{x:.2f}'
                elif x >= 0.01:
                    return f'{x:.3f}'
                else:
                    return f'{x:.4f}'
            tick_vals.append(min_val)
            tick_text.append(format_number(min_val))
            for i in range(int(log_min), int(log_max) + 1):
                current = 10**i
                if min_val <= current <= max_val:
                    if current not in tick_vals:
                        tick_vals.append(current)
                        tick_text.append(format_number(current))
                if 10 * current <= max_val and 10 * current >= min_val:
                    if (10 * current) not in tick_vals:
                        tick_vals.append(10 * current)
                        tick_text.append(format_number(10 * current))
            if max_val not in tick_vals:
                tick_vals.append(max_val)
                tick_text.append(format_number(max_val))
            tick_vals, tick_text = zip(*sorted(zip(tick_vals, tick_text)))
            tick_vals = list(tick_vals)
            tick_text = list(tick_text)
        else:
            color_values = valid_samples[primary_element]
            color_bar_title = primary_element
            min_val = valid_samples[primary_element].min()
            max_val = valid_samples[primary_element].max()
            tick_vals = np.linspace(min_val, max_val, 6)
            tick_text = [f'{v:.3f}' for v in tick_vals]
        custom_colorscale = 'rdylbu'

    hover_text = []
    for _, row in valid_samples.iterrows():
        hover_info = [
            f"<b>Hole ID:</b> {row['HOLE_ID']}",
            f"<b>{primary_element}:</b> {row[primary_element]:.2f}",
            f"<b>From:</b> {row['FROM']:.2f}",
            f"<b>To:</b> {row['TO']:.2f}"
        ]
        if 'LITHO' in valid_samples.columns:
            hover_info.append(f"<b>Lithology:</b> {row['LITHO']}")
        if 'Cluster' in valid_samples.columns:
            hover_info.append(f"<b>Cluster:</b> {row['Cluster']}")
        hover_text.append("<br>".join(hover_info))

    fig.add_trace(go.Scatter3d(
        x=valid_samples['x'] + x_offset,
        y=valid_samples['y'],
        z=valid_samples['z'],
        mode='markers',
        marker=dict(
            size=8,  # Increased marker size from 5 to 8
            color=color_values,
            colorscale=custom_colorscale,
            reversescale=True if color_by != 'cluster' else False,
            showscale=True,
            symbol='square',
            colorbar=dict(
                title=color_bar_title,
                len=0.75,
                titleside='right',
                ticks='outside',
                ticklen=5,
                x=0.95,
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text
            ),
            cmin=min_val,
            cmax=max_val,
            cauto=False
        ),
        hovertemplate="%{text}<br>" +
                      "<b>X:</b> %{x:.2f}<br>" +
                      "<b>Y:</b> %{y:.2f}<br>" +
                      "<b>Z:</b> %{z:.2f}<extra></extra>",
        text=hover_text,
        name='Samples',
        showlegend=False
    ))

def add_lithology_visualisation(fig, viz_litho_df, viz_mode, selected_lithos=None, litho_dict=None, x_offset=0):
    """Add lithology visualisation to the figure"""
    if viz_litho_df is None or viz_litho_df.empty:
        st.warning("No lithology data available for display.")
        return
    unique_lithos = viz_litho_df['LITHO'].unique()
    color_palette = px.colors.qualitative.Alphabet
    litho_colors = [color_palette[i % len(color_palette)] for i in range(len(unique_lithos))]
    litho_color_map = dict(zip(unique_lithos, litho_colors))
    legend_added = set()
    square_size = 8  # Increased marker size from 5 to 8
    for hole_id in viz_litho_df['HOLE_ID'].unique():
        hole_data = viz_litho_df[viz_litho_df['HOLE_ID'] == hole_id].sort_values('FROM')
        collar = hole_data.iloc[0]
        collar_x, collar_y, collar_z = collar['EASTING'], collar['NORTHING'], collar['ELEVATION']
        for _, interval in hole_data.iterrows():
            litho_code = interval['LITHO']
            litho_desc = litho_dict.get(litho_code, "") if litho_dict else ""
            legend_name = f"{litho_code} {litho_desc}".strip() if litho_dict else litho_code
            show_legend = legend_name not in legend_added
            if show_legend:
                legend_added.add(legend_name)
            depths = np.linspace(interval['FROM'], interval['TO'], 
                                 num=max(2, int((interval['TO'] - interval['FROM']) / 2)))
            azimuth_rad = np.radians(90 - interval['AZIMUTH'])
            dip_rad = np.radians(-interval['DIP'])
            x_values = collar_x + depths * np.cos(dip_rad) * np.cos(azimuth_rad) + x_offset
            y_values = collar_y + depths * np.cos(dip_rad) * np.sin(azimuth_rad)
            z_values = collar_z - depths * np.sin(dip_rad)
            fig.add_trace(go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='markers',
                marker=dict(
                    symbol='square',
                    size=square_size,
                    color=litho_color_map[litho_code],
                ),
                name=legend_name,
                legendgroup=legend_name,
                hovertemplate=(
                    f"<b>Hole ID:</b> {hole_id}<br>" +
                    f"<b>Lithology:</b> {legend_name}<br>" +
                    "<b>From:</b> {:.2f}<br>".format(interval['FROM']) +
                    "<b>To:</b> {:.2f}<br>".format(interval['TO']) +
                    "<b>X</b>: %{x:.2f}<br>" +
                    "<b>Y</b>: %{y:.2f}<br>" +
                    "<b>Z</b>: %{z:.2f}<br>"
                ),
                showlegend=show_legend
            ))

def add_collar_points(fig, collar_df, x_offset=0):
    """Add collar points to the figure"""
    fig.add_trace(go.Scatter3d(
        x=collar_df['EASTING'] + x_offset,
        y=collar_df['NORTHING'],
        z=collar_df['ELEVATION'],
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            symbol='circle'
        ),
        name='Collar Points',
        hovertemplate=(
            "<b>Hole ID:</b> %{customdata}<br>" +
            "<b>Easting:</b> %{x:.2f}<br>" +
            "<b>Northing:</b> %{y:.2f}<br>" +
            "<b>Elevation:</b> %{z:.2f}<br>"
        ),
        customdata=collar_df['HOLE_ID']
    ))

def update_figure_layout(fig, vertical_exaggeration=1.0):
    """Update figure layout with proper aspect ratios"""
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            x_min = min(x_min, min(trace.x))
            x_max = max(x_max, max(trace.x))
        if hasattr(trace, 'y') and trace.y is not None:
            y_min = min(y_min, min(trace.y))
            y_max = max(y_max, max(trace.y))
        if hasattr(trace, 'z') and trace.z is not None:
            z_min = min(z_min, min(trace.z))
            z_max = max(z_max, max(trace.z))

    x_diff = x_max - x_min
    y_diff = y_max - y_min
    z_diff = z_max - z_min
    max_diff = max(x_diff, y_diff, z_diff * vertical_exaggeration)

    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(
                x=x_diff / max_diff,
                y=y_diff / max_diff,
                z=(z_diff * vertical_exaggeration) / max_diff
            ),
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title="Elevation"
        ),
        width=1800,  # Increased width from 1400 to 1800
        height=1200,  # Increased height from 1000 to 1200
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="true",
        legend=dict(
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            title="Lithology"
        ),
        legend2=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            title="Clusters"
        )
    )
    fig.update_scenes(
        xaxis_range=[x_min, x_max],
        yaxis_range=[y_min, y_max],
        zaxis_range=[z_min, z_max]
    )

def show_statistical_analysis(merged_df, primary_element, use_log_scale):
    """Show basic stats and histogram"""
    st.header("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        stats_dict = {
            'Statistic': [
                'Count', 'Mean', 'Median', 'Std Dev', 'CV', 
                'Min', 'Q1', 'Q3', 'Max', 'Skewness', 'Kurtosis'
            ],
            'Value': [
                len(merged_df[primary_element]),
                merged_df[primary_element].mean(),
                merged_df[primary_element].median(),
                merged_df[primary_element].std(),
                merged_df[primary_element].std() / merged_df[primary_element].mean(),
                merged_df[primary_element].min(),
                merged_df[primary_element].quantile(0.25),
                merged_df[primary_element].quantile(0.75),
                merged_df[primary_element].max(),
                merged_df[primary_element].skew(),
                merged_df[primary_element].kurtosis()
            ]
        }
        stats_df = pd.DataFrame(stats_dict)
        stats_df['Value'] = stats_df['Value'].round(2)
        
        # Make the dataframe wider and taller - increased height to 450
        st.dataframe(stats_df.set_index('Statistic'), width=400, height=420)
        
    with col2:
        st.subheader("Histogram")
        if use_log_scale and merged_df[primary_element].min() > 0:
            log_data = np.log10(merged_df[primary_element])
            fig = px.histogram(log_data, nbins=30, title=f"{primary_element} Distribution (log scale)")
            min_val = merged_df[primary_element].min()
            q1_val = merged_df[primary_element].quantile(0.25)
            median_val = merged_df[primary_element].median()
            q3_val = merged_df[primary_element].quantile(0.75)
            max_val = merged_df[primary_element].max()
            tick_vals = [np.log10(val) for val in [min_val, q1_val, median_val, q3_val, max_val]]
            tick_text = [f'{val:,.2f}' for val in [min_val, q1_val, median_val, q3_val, max_val]]
            fig.update_xaxes(
                title_text=primary_element,
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[np.log10(min_val), np.log10(max_val)]
            )
        else:
            fig = px.histogram(merged_df[primary_element], nbins=30, title=f"{primary_element} Distribution")
            min_val = merged_df[primary_element].min()
            q1_val = merged_df[primary_element].quantile(0.25)
            median_val = merged_df[primary_element].median()
            q3_val = merged_df[primary_element].quantile(0.75)
            max_val = merged_df[primary_element].max()
            fig.update_xaxes(
                title_text=primary_element,
                tickmode='array',
                tickvals=[min_val, q1_val, median_val, q3_val, max_val],
                ticktext=[f'{v:.2f}' for v in [min_val, q1_val, median_val, q3_val, max_val]]
            )
        st.plotly_chart(fig, use_container_width=True)
        
def calculate_significant_intervals(df, element, cutoff, min_length, max_internal_waste, litho_dict=None):
    """Find significant intervals based on cutoff grade"""
    results = []
    for hole_id, hole_data in df.groupby('HOLE_ID'):
        hole_data = hole_data.sort_values('FROM')
        current_interval = {
            'start_depth': None, 
            'end_depth': None, 
            'grades': [], 
            'lengths': [], 
            'lithos': [], 
            'waste_lengths': [], 
            'last_significant_to': None
        }
        for idx, row in hole_data.iterrows():
            try:
                interval_length = float(row['TO']) - float(row['FROM'])
                current_grade = float(row[element])
                if interval_length <= 0 or pd.isna(current_grade):
                    continue
                if current_grade >= cutoff:
                    if (current_interval['start_depth'] is None or
                       (current_interval['last_significant_to'] is not None and 
                        row['FROM'] - current_interval['last_significant_to'] <= max_internal_waste)):
                        if current_interval['start_depth'] is None:
                            current_interval['start_depth'] = row['FROM']
                        if (current_interval['last_significant_to'] is not None and 
                            row['FROM'] > current_interval['last_significant_to']):
                            waste_length = row['FROM'] - current_interval['last_significant_to']
                            current_interval['waste_lengths'].append(waste_length)
                        current_interval['grades'].append(current_grade)
                        current_interval['lengths'].append(interval_length)
                        if 'LITHO' in row:
                            current_interval['lithos'].append(row['LITHO'])
                        current_interval['end_depth'] = row['TO']
                        current_interval['last_significant_to'] = row['TO']
                    else:
                        if current_interval['start_depth'] is not None:
                            total_length = sum(current_interval['lengths'])
                            total_waste = sum(current_interval['waste_lengths'])
                            if total_length >= min_length:
                                weighted_grade = np.average(
                                    current_interval['grades'],
                                    weights=current_interval['lengths']
                                )
                                interval_dict = {
                                    'HOLE_ID': hole_id,
                                    'FROM': current_interval['start_depth'],
                                    'TO': current_interval['end_depth'],
                                    'LENGTH': total_length + total_waste,
                                    f'{element}_GRADE': weighted_grade,
                                    'INTERNAL_WASTE': total_waste,
                                    'LITHOLOGY': ' / '.join(set(current_interval['lithos']))
                                }
                                if litho_dict is not None and current_interval['lithos']:
                                    descriptions = []
                                    for lith in set(current_interval['lithos']):
                                        if desc := litho_dict.get(lith):
                                            descriptions.append(desc)
                                    if descriptions:
                                        interval_dict['DESCRIPTION'] = ' / '.join(descriptions)
                                results.append(interval_dict)
                        current_interval = {
                            'start_depth': row['FROM'],
                            'end_depth': row['TO'],
                            'grades': [current_grade],
                            'lengths': [interval_length],
                            'lithos': ['LITHO' in row and row['LITHO'] or []],
                            'waste_lengths': [],
                            'last_significant_to': row['TO']
                        }
                elif current_interval['start_depth'] is not None:
                    if row['FROM'] - current_interval['last_significant_to'] > max_internal_waste:
                        total_length = sum(current_interval['lengths'])
                        total_waste = sum(current_interval['waste_lengths'])
                        if total_length >= min_length:
                            weighted_grade = np.average(
                                current_interval['grades'],
                                weights=current_interval['lengths']
                            )
                            interval_dict = {
                                'HOLE_ID': hole_id,
                                'FROM': current_interval['start_depth'],
                                'TO': current_interval['end_depth'],
                                'LENGTH': total_length + total_waste,
                                f'{element}_GRADE': weighted_grade,
                                'INTERNAL_WASTE': total_waste,
                                'LITHOLOGY': ' / '.join(set(current_interval['lithos']))
                            }
                            if litho_dict is not None and current_interval['lithos']:
                                descriptions = []
                                for lith in set(current_interval['lithos']):
                                    if desc := litho_dict.get(lith):
                                        descriptions.append(desc)
                                if descriptions:
                                    interval_dict['DESCRIPTION'] = ' / '.join(descriptions)
                            results.append(interval_dict)
                        current_interval = {
                            'start_depth': None,
                            'end_depth': None,
                            'grades': [],
                            'lengths': [],
                            'lithos': [],
                            'waste_lengths': [],
                            'last_significant_to': None
                        }
            except (ValueError, TypeError):
                continue
        if current_interval['start_depth'] is not None:
            total_length = sum(current_interval['lengths'])
            total_waste = sum(current_interval['waste_lengths'])
            if total_length >= min_length:
                weighted_grade = np.average(
                    current_interval['grades'],
                    weights=current_interval['lengths']
                )
                interval_dict = {
                    'HOLE_ID': hole_id,
                    'FROM': current_interval['start_depth'],
                    'TO': current_interval['end_depth'],
                    'LENGTH': total_length + total_waste,
                    f'{element}_GRADE': weighted_grade,
                    'INTERNAL_WASTE': total_waste,
                    'LITHOLOGY': ' / '.join(set(current_interval['lithos']))
                }
                if litho_dict is not None and current_interval['lithos']:
                    descriptions = []
                    for lith in set(current_interval['lithos']):
                        if desc := litho_dict.get(lith):
                            descriptions.append(desc)
                    if descriptions:
                        interval_dict['DESCRIPTION'] = ' / '.join(descriptions)
                results.append(interval_dict)
    return pd.DataFrame(results)

def create_cluster_visualisation(merged_df, viz_df, collar_df, primary_element, use_log_scale, vertical_exaggeration=1.0):
    """Create 3D visualisation of clusters with the same options as in the visuals tab"""
    fig = go.Figure()
    
    # Add cluster visualisation
    if 'Cluster' in viz_df.columns:
        # Add cluster points
        for cluster in sorted(viz_df['Cluster'].unique()):
            if cluster >= 0:  # Skip invalid clusters
                cluster_data = viz_df[viz_df['Cluster'] == cluster]
                if not cluster_data.empty:
                    hover_text = []
                    for _, row in cluster_data.iterrows():
                        info = [
                            f"<b>Hole ID:</b> {row['HOLE_ID']}",
                            f"<b>Cluster:</b> {cluster}"
                        ]
                        if primary_element:
                            info.append(f"<b>{primary_element}:</b> {row[primary_element]:.2f}")
                        info.append(f"<b>From:</b> {row['FROM']:.2f}")
                        info.append(f"<b>To:</b> {row['TO']:.2f}")
                        if 'LITHO' in row:
                            info.append(f"<b>Lithology:</b> {row['LITHO']}")
                        hover_text.append("<br>".join(info))
                    fig.add_trace(go.Scatter3d(
                        x=cluster_data['x'],
                        y=cluster_data['y'],
                        z=cluster_data['z'],
                        mode='markers',
                        marker=dict(
                            size=8,  # Increased marker size from 5 to 8
                            color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]
                        ),
                        name=f'Cluster {cluster}',
                        hovertemplate="%{text}<br>" +
                                    "<b>X:</b> %{x:.2f}<br>" +
                                    "<b>Y:</b> %{y:.2f}<br>" +
                                    "<b>Z:</b> %{z:.2f}<extra></extra>",
                        text=hover_text
                    ))
        
        # Add drill hole lines
        for hole in viz_df['HOLE_ID'].unique():
            hole_data = viz_df[viz_df['HOLE_ID'] == hole]
            collar_point = collar_df[collar_df['HOLE_ID'] == hole].iloc[0]
            x_line = [collar_point['EASTING']] + hole_data['x'].tolist()
            y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
            z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        
        # Add collar points
        add_collar_points(fig, collar_df)
    
    # Update layout with proper aspect ratios
    update_figure_layout(fig, vertical_exaggeration)
    
    return fig

# Function to apply filters to dataframe
def apply_filters(df, selected_holes, selected_lithos, primary_element=None, min_cutoff=None, max_cutoff=None):
    filtered_df = df.copy()
    
    # Apply hole filter
    if selected_holes:
        filtered_df = filtered_df[filtered_df['HOLE_ID'].isin(selected_holes)]
    
    # Apply lithology filter
    if selected_lithos and 'LITHO' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['LITHO'].isin(selected_lithos)]
    
    # Apply element cutoff filter
    if primary_element and min_cutoff is not None and max_cutoff is not None:
        filtered_df = filtered_df[
            (filtered_df[primary_element] >= min_cutoff) &
            (filtered_df[primary_element] <= max_cutoff)
        ]
    
    return filtered_df

def run_clustering_analysis(cluster_df, cluster_features, use_pca, use_log_transform, n_components, max_clusters):
    if len(cluster_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
        return None, None, None
    elif cluster_df.empty:
        st.warning("No data available for clustering after applying filters.")
        return None, None, None
    else:
        X = cluster_df[cluster_features]
        if use_pca:
            X_pca, pca, scaler = perform_pca_analysis(X, n_components, use_log_transform)
            st.subheader("PCA Scree Plot")
            st.plotly_chart(plot_scree(explained_variance_ratio=pca.explained_variance_ratio_, is_pca=True))
            n_components = st.slider("Select final number of components", 2, len(cluster_features), n_components)
            if n_components != pca.n_components_:
                X_pca, pca, scaler = perform_pca_analysis(X, n_components, use_log_transform)
            st.plotly_chart(plot_pca_biplot(X_pca, pca, cluster_features))
            st.subheader("Principal Component Loadings")
            eigenvector_fig = plot_eigenvectors(pca, cluster_features)
            st.plotly_chart(eigenvector_fig)
            wcss = []
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                kmeans.fit(X_pca)
                wcss.append(kmeans.inertia_)
            return X_pca, None, wcss
        else:
            X_scaled, scaler, wcss = perform_clustering(cluster_df, cluster_features, max_clusters, use_log_transform)
            return X_scaled, scaler, wcss

def process_and_merge_data(collar_df, assay_df, litho_df, element_cols, composite_enabled, composite_length):
    merged_df = None
    viz_litho_df = None
    
    if collar_df is not None:
        if assay_df is not None:
            # Apply compositing if needed
            if composite_enabled and element_cols:
                assay_df = composite_geochemical_data(assay_df, element_cols, composite_length)

            merged_df = pd.merge(collar_df, assay_df, on='HOLE_ID', how='inner')
            if 'FROM' in merged_df.columns and 'TO' in merged_df.columns:
                merged_df['MIDPOINT'] = (merged_df['FROM'] + merged_df['TO']) / 2
                merged_df['AZIMUTH_RAD'] = np.radians(90 - merged_df['AZIMUTH'])
                merged_df['DIP_RAD'] = np.radians(merged_df['DIP'])
                merged_df['dx'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.cos(merged_df['AZIMUTH_RAD'])
                merged_df['dy'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.sin(merged_df['AZIMUTH_RAD'])
                merged_df['dz'] = merged_df['MIDPOINT'] * np.sin(merged_df['DIP_RAD'])
                merged_df['x'] = merged_df['EASTING'] + merged_df['dx']
                merged_df['y'] = merged_df['NORTHING'] + merged_df['dy']
                merged_df['z'] = merged_df['ELEVATION'] + merged_df['dz']

        if litho_df is not None:
            if merged_df is None:
                merged_df = pd.merge(collar_df, litho_df, on='HOLE_ID', how='inner')
                merged_df['MIDPOINT'] = (merged_df['FROM'] + merged_df['TO']) / 2
                merged_df['AZIMUTH_RAD'] = np.radians(90 - merged_df['AZIMUTH'])
                merged_df['DIP_RAD'] = np.radians(merged_df['DIP'])
                merged_df['dx'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.cos(merged_df['AZIMUTH_RAD'])
                merged_df['dy'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.sin(merged_df['AZIMUTH_RAD'])
                merged_df['dz'] = merged_df['MIDPOINT'] * np.sin(merged_df['DIP_RAD'])
                merged_df['x'] = merged_df['EASTING'] + merged_df['dx']
                merged_df['y'] = merged_df['NORTHING'] + merged_df['dy']
                merged_df['z'] = merged_df['ELEVATION'] + merged_df['dz']
            else:
                merged_df = pd.merge_asof(
                    merged_df.sort_values('FROM'),
                    litho_df.sort_values('FROM'),
                    by='HOLE_ID',
                    on='FROM',
                    direction='nearest'
                )
                merged_df = merged_df.rename(columns={'TO_x': 'TO'})
                if 'TO_y' in merged_df.columns:
                    merged_df.drop('TO_y', axis=1, inplace=True)

            # Create viz litho dataframe
            viz_litho_df = litho_df.copy()
            viz_litho_df = pd.merge(viz_litho_df, collar_df[['HOLE_ID','EASTING','NORTHING','ELEVATION','DIP','AZIMUTH']], on='HOLE_ID')
            viz_litho_df['MIDPOINT'] = (viz_litho_df['FROM'] + viz_litho_df['TO']) / 2
            viz_litho_df['AZIMUTH_RAD'] = np.radians(90 - viz_litho_df['AZIMUTH'])
            viz_litho_df['DIP_RAD'] = np.radians(viz_litho_df['DIP'])
            viz_litho_df['dx'] = viz_litho_df['MIDPOINT'] * np.cos(viz_litho_df['DIP_RAD']) * np.cos(viz_litho_df['AZIMUTH_RAD'])
            viz_litho_df['dy'] = viz_litho_df['MIDPOINT'] * np.cos(viz_litho_df['DIP_RAD']) * np.sin(viz_litho_df['AZIMUTH_RAD'])
            viz_litho_df['dz'] = viz_litho_df['MIDPOINT'] * np.sin(viz_litho_df['DIP_RAD'])
            viz_litho_df['x'] = viz_litho_df['EASTING'] + viz_litho_df['dx']
            viz_litho_df['y'] = viz_litho_df['NORTHING'] + viz_litho_df['dy']
            viz_litho_df['z'] = viz_litho_df['ELEVATION'] + viz_litho_df['dz']
    
    return merged_df, viz_litho_df

# MAIN APP
# Create tabs for different sections
tab_data, tab_viz, tab_stats, tab_clustering, tab_download = st.tabs([
    "Data Loading", "Visualisations", "Statistics", "Clustering", "Export Data"
])

with tab_data:
    st.header("Data Loading")
    
    file_format = st.radio(
        "Select CSV File Format",
        ("Standard CSV (Headers in row 1)", "Geological Survey Format (Headers in H1000)")
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        collar_file = st.file_uploader("Upload Collar File (CSV)", type=['csv'])
    with col2:
        assay_file = st.file_uploader("Upload Assay File (CSV)", type=['csv'])
    with col3:
        litho_file = st.file_uploader("Upload Lithology File (CSV)", type=['csv'])
        litho_dict_file = st.file_uploader("Upload Lithology Dictionary File (CSV)", type=['csv'])

    if (collar_file is not None and collar_file != st.session_state.previous_collar_file) or \
       (assay_file is not None and assay_file != st.session_state.previous_assay_file):
        st.session_state.X_scaled = None
        st.session_state.scaler = None
        st.session_state.wcss = None
        st.session_state.n_clusters = 3
        st.session_state.selected_cluster_features = None
        st.session_state.previous_collar_file = collar_file
        st.session_state.previous_assay_file = assay_file

    valid_data_combinations = False
    if collar_file:
        if assay_file and not litho_file:
            valid_data_combinations = True
            st.session_state.analysis_mode = "collar_assay"
        elif litho_file and not assay_file:
            valid_data_combinations = True
            st.session_state.analysis_mode = "collar_litho"
        elif assay_file and litho_file:
            valid_data_combinations = True
            st.session_state.analysis_mode = "all"
    else:
        st.warning("Collar file is required.")

    if valid_data_combinations:
        st.session_state.collar_df = process_collar_data(collar_file, file_format)
        
        if st.session_state.collar_df is not None:
            if st.session_state.analysis_mode in ["collar_assay", "all"]:
                assay_df, st.session_state.element_cols = process_assay_data(assay_file, file_format)
                
                # Compositing options
                st.sidebar.markdown("<h1 style='font-size: 28px;'>OPTIONS</h1>", unsafe_allow_html=True)

                st.sidebar.header("Compositing Options")
                composite_enabled = st.sidebar.checkbox("Composite geochemical data")
                composite_length = 1.0
                if composite_enabled:
                    composite_length = st.sidebar.slider("Composite Interval (m)", min_value=1, max_value=10, value=2)
            else:
                assay_df = None
                
            litho_df = None
            if st.session_state.analysis_mode in ["collar_litho", "all"]:
                litho_df = process_litho_data(litho_file, file_format)
                
                # Process lithology dictionary
                st.session_state.litho_dict = None
                if litho_dict_file:
                    st.session_state.litho_dict = process_litho_dict(litho_dict_file, file_format)
            
            # Process and merge data
            st.session_state.merged_df, st.session_state.viz_litho_df = process_and_merge_data(
                st.session_state.collar_df, 
                assay_df, 
                litho_df, 
                st.session_state.element_cols, 
                composite_enabled, 
                composite_length
            )
            
            if st.session_state.merged_df is not None:
                st.success("Data loaded and processed successfully!")
                st.write("Preview of processed data:")
                st.write(st.session_state.merged_df.head())
            else:
                st.error("Failed to process data. Please check your inputs.")

with tab_viz:
    st.header("3D Visualisation")
    
    if st.session_state.merged_df is not None:
        original_merged_df = st.session_state.merged_df.copy()
        st.session_state.viz_df = st.session_state.merged_df.copy()
        
        # Filter options
        st.sidebar.header("Filter Options")
        st.session_state.apply_filters_globally = st.sidebar.checkbox(
            "Apply filters to all analyses (not just visualisation)", 
            value=st.session_state.apply_filters_globally
        )
        
        selected_holes = []
        selected_lithos = []
        all_holes = sorted(st.session_state.merged_df['HOLE_ID'].unique())
        selected_holes = st.sidebar.multiselect("Select holes to display (empty for all)", options=all_holes)
        use_log_scale = st.sidebar.checkbox("Use log scale", value=True, key="main_log_scale")

        primary_element = None
        min_cutoff = None
        max_cutoff = None
        
        if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
            primary_element = st.sidebar.selectbox("Select element for analysis:", st.session_state.element_cols)
            min_cutoff, max_cutoff = st.sidebar.slider(
                f"{primary_element} cutoff range",
                min_value=float(st.session_state.merged_df[primary_element].min()),
                max_value=float(st.session_state.merged_df[primary_element].max()),
                value=(float(st.session_state.merged_df[primary_element].min()), float(st.session_state.merged_df[primary_element].max())),
                step=0.1
            )

        if 'LITHO' in st.session_state.merged_df.columns:
            all_lithos = sorted(st.session_state.merged_df['LITHO'].unique())
            selected_lithos = st.sidebar.multiselect(
                "Select lithologies to display (empty for all)",
                options=all_lithos
            )
        
        # Apply filters to visualisation dataframe
        st.session_state.viz_df = apply_filters(st.session_state.viz_df, selected_holes, selected_lithos, primary_element, min_cutoff, max_cutoff)
        
        # Apply filters to viz_litho_df if it exists
        viz_litho_df = st.session_state.viz_litho_df
        if viz_litho_df is not None:
            if selected_holes:
                viz_litho_df = viz_litho_df[viz_litho_df['HOLE_ID'].isin(selected_holes)]
            if selected_lithos and 'LITHO' in viz_litho_df.columns:
                viz_litho_df = viz_litho_df[viz_litho_df['LITHO'].isin(selected_lithos)]
        
        # Apply filters to collar_df for visualisation
        viz_collar_df = st.session_state.collar_df.copy()
        if selected_holes:
            viz_collar_df = viz_collar_df[viz_collar_df['HOLE_ID'].isin(selected_holes)]
        
        # Apply filters globally if checkbox is selected
        if st.session_state.apply_filters_globally:
            st.session_state.merged_df = apply_filters(st.session_state.merged_df, selected_holes, selected_lithos, primary_element, min_cutoff, max_cutoff)
            
            # Reset session state for clustering if filters changed
            if 'previous_filter_state' not in st.session_state or st.session_state.previous_filter_state != (tuple(selected_holes), tuple(selected_lithos), primary_element, min_cutoff, max_cutoff):
                st.session_state.X_scaled = None
                st.session_state.scaler = None
                st.session_state.wcss = None
                st.session_state.previous_filter_state = (tuple(selected_holes), tuple(selected_lithos), primary_element, min_cutoff, max_cutoff)
        else:
            # Use original data for analysis if global filters not applied
            st.session_state.merged_df = original_merged_df.copy()

        vertical_exaggeration = st.slider(
            "Vertical Exaggeration", 
            min_value=1.0, 
            max_value=10.0, 
            value=1.0, 
            step=0.1
        )

        viz_options = []
        if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
            viz_options.append("Grade")
        if 'LITHO' in st.session_state.merged_df.columns:
            viz_options.append("Lithology")
        if 'Cluster' in st.session_state.merged_df.columns:
            viz_options.append("Clusters")
            
        selected_viz = st.multiselect(
            "Select visualisation types. (If multiple are selected, they will be offset.)",
            viz_options,
            default=viz_options[0] if viz_options else None
        )

        if selected_viz:
            fig = go.Figure()
            offsets = {"Grade": 0, "Clusters": 20, "Lithology": -20}
            for viz_type in selected_viz:
                if viz_type == "Grade" and st.session_state.analysis_mode in ["collar_assay", "all"] and 'primary_element' in locals():
                    if not st.session_state.viz_df.empty:
                        add_grade_visualisation(fig, st.session_state.viz_df, primary_element, use_log_scale, "Combined", color_by='grade', x_offset=offsets["Grade"])
                        for hole in st.session_state.viz_df['HOLE_ID'].unique():
                            hole_data = st.session_state.viz_df[st.session_state.viz_df['HOLE_ID'] == hole]
                            collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole].iloc[0]
                            x_line = [collar_point['EASTING'] + offsets["Grade"]] + (hole_data['x'] + offsets["Grade"]).tolist()
                            y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                            z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                            fig.add_trace(go.Scatter3d(
                                x=x_line, y=y_line, z=z_line,
                                mode='lines',
                                line=dict(color='gray', width=1),
                                showlegend=False
                            ))
                        add_collar_points(fig, viz_collar_df, x_offset=offsets["Grade"])
                    else:
                        st.warning("No data available for Grade visualisation after applying filters.")

                if viz_type == "Lithology" and viz_litho_df is not None:
                    if not viz_litho_df.empty:
                        add_lithology_visualisation(fig, viz_litho_df, "Combined", selected_lithos, st.session_state.litho_dict, x_offset=offsets["Lithology"])
                        for hole in viz_litho_df['HOLE_ID'].unique():
                            hole_data = viz_litho_df[viz_litho_df['HOLE_ID'] == hole]
                            collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole].iloc[0]
                            x_line = [collar_point['EASTING'] + offsets["Lithology"]] + (hole_data['x'] + offsets["Lithology"]).tolist()
                            y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                            z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                            fig.add_trace(go.Scatter3d(
                                x=x_line, y=y_line, z=z_line,
                                mode='lines',
                                line=dict(color='gray', width=1),
                                showlegend=False
                            ))
                        add_collar_points(fig, viz_collar_df, x_offset=offsets["Lithology"])
                    else:
                        st.warning("No data available for Lithology visualisation after applying filters.")
                        
                if viz_type == "Clusters" and 'Cluster' in st.session_state.viz_df.columns:
                    # Filter to only include rows with valid cluster assignments
                    cluster_viz_df = st.session_state.viz_df[st.session_state.viz_df['Cluster'] >= 0].copy()
                    
                    if not cluster_viz_df.empty:
                        # Add cluster visualisation
                        for cluster in sorted(cluster_viz_df['Cluster'].unique()):
                            cluster_data = cluster_viz_df[cluster_viz_df['Cluster'] == cluster]
                            hover_text = []
                            for _, row in cluster_data.iterrows():
                                info = [
                                    f"<b>Hole ID:</b> {row['HOLE_ID']}",
                                    f"<b>Cluster:</b> {cluster}"
                                ]
                                if 'primary_element' in locals():
                                    info.append(f"<b>{primary_element}:</b> {row[primary_element]:.2f}")
                                info.append(f"<b>From:</b> {row['FROM']:.2f}")
                                info.append(f"<b>To:</b> {row['TO']:.2f}")
                                if 'LITHO' in row:
                                    info.append(f"<b>Lithology:</b> {row['LITHO']}")
                                hover_text.append("<br>".join(info))
                            
                            fig.add_trace(go.Scatter3d(
                                x=cluster_data['x'] + offsets["Clusters"],
                                y=cluster_data['y'],
                                z=cluster_data['z'],
                                mode='markers',
                                marker=dict(
                                    size=8,  # Increased marker size from 5 to 8
                                    color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]
                                ),
                                name=f'Cluster {cluster}',
                                hovertemplate="%{text}<br>" +
                                            "<b>X:</b> %{x:.2f}<br>" +
                                            "<b>Y:</b> %{y:.2f}<br>" +
                                            "<b>Z:</b> %{z:.2f}<extra></extra>",
                                text=hover_text
                            ))
                        
                        # Add drill hole lines and collar points as you already have
                    else:
                        st.warning("No cluster data available for visualisation after applying filters.")

            update_figure_layout(fig, vertical_exaggeration)
            st.plotly_chart(fig)
        else:
            st.warning("Please select at least one visualisation type.")
            
        if st.session_state.analysis_mode in ["collar_assay", "all"] and 'primary_element' in locals():
            if not st.session_state.merged_df.empty:
                create_swath_plots(st.session_state.merged_df, primary_element, use_log_scale)
    else:
        st.warning("Please load data in the Data Loading tab first.")

with tab_stats:
    st.header("Statistical Analysis")
    
    if st.session_state.merged_df is not None:
        if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
            if not st.session_state.merged_df.empty:
                # Select element for analysis
                primary_element = st.selectbox("Select element for statistical analysis:", 
                                               st.session_state.element_cols,
                                               key="stats_primary_element")
                use_log_scale = st.checkbox("Use log scale for visualisations", value=True, key="stats_log_scale")
                
                # Show basic statistics
                show_statistical_analysis(st.session_state.merged_df, primary_element, use_log_scale)
                
                # Correlation analysis
                st.header("Correlation Analysis")
                correlation_matrix = st.session_state.merged_df[st.session_state.element_cols].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix,
                    x=st.session_state.element_cols,
                    y=st.session_state.element_cols,
                    text=np.round(correlation_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu',
                    zmid=0
                ))
                fig.update_layout(title="Correlation Matrix", height=500, width=500)
                st.plotly_chart(fig)

                # Element selection for scatter plots
                st.subheader("Element selection for Correlation Analysis and Scatter Diagrams")
                scatter_elements = st.multiselect(
                    "Select elements for scatter plot (minimum 2)",
                    st.session_state.element_cols,
                    default=st.session_state.element_cols[:min(3, len(st.session_state.element_cols))]
                )
                if len(scatter_elements) >= 2:
                    # Create correlation matrix for selected elements
                    st.subheader("Selected Elements Correlation Matrix")
                    corr_stats = pd.DataFrame(
                        [[st.session_state.merged_df[e1].corr(st.session_state.merged_df[e2]) for e2 in scatter_elements]
                            for e1 in scatter_elements],
                        columns=scatter_elements,
                        index=scatter_elements
                    )

                    # Plotly correlation matrix
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_stats,
                        x=scatter_elements,
                        y=scatter_elements,
                        text=np.round(corr_stats, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorscale='RdBu',
                        zmid=0
                    ))

                    # Update layout                 
                    fig.update_layout(
                        height=max(400, len(scatter_elements) * 40),  # Dynamic height based on number of elements
                        width=max(500, len(scatter_elements) * 50),   # Dynamic width based on number of elements
                        xaxis=dict(tickangle=-45),
                        margin=dict(l=50, r=50, t=50, b=50)
                    )

                    st.plotly_chart(fig)
                    st.subheader("Selected Elements Scatter Diagrams")
                    # Generate scatter plots
                    pairs = [(i, j) for i in scatter_elements for j in scatter_elements if i < j]
                    n_pairs = len(pairs)
                    n_cols = min(3, n_pairs)
                    n_rows = (n_pairs + n_cols - 1) // n_cols
                    fig = make_subplots(rows=n_rows, cols=n_cols)
                    def format_tick_label(value):
                        if value >= 1:
                            return f'{value:.0f}'
                        elif value >= 0.1:
                            return f'{value:.2f}'
                        elif value >= 0.01:
                            return f'{value:.3f}'
                        else:
                            return f'{value:.4f}'
                    idx = 0
                    for elem1, elem2 in pairs:
                        row = idx // n_cols + 1
                        col = idx % n_cols + 1
                        scatter = go.Scatter(
                            x=st.session_state.merged_df[elem1], y=st.session_state.merged_df[elem2],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=st.session_state.merged_df[primary_element],
                                colorscale='Viridis',
                                showscale=True if idx == 0 else False,
                                colorbar=dict(title=primary_element) if idx == 0 else None,
                                opacity=0.7
                            ),
                            name=f'{elem1} vs {elem2}'
                        )
                        fig.add_trace(scatter, row=row, col=col)
                        fig.update_xaxes(title_text=elem1, row=row, col=col)
                        fig.update_yaxes(title_text=elem2, row=row, col=col)
                        if use_log_scale:
                            def log_tick_values(data):
                                min_val = max(data.min(), 1e-10)
                                max_val = data.max()
                                return np.logspace(np.log10(min_val), np.log10(max_val), num=6)
                            x_ticks = log_tick_values(st.session_state.merged_df[elem1])
                            y_ticks = log_tick_values(st.session_state.merged_df[elem2])
                            fig.update_xaxes(
                                type="log",
                                tickmode='array',
                                tickvals=x_ticks,
                                ticktext=[format_tick_label(x) for x in x_ticks],
                                row=row, col=col
                            )
                            fig.update_yaxes(
                                type="log",
                                tickmode='array',
                                tickvals=y_ticks,
                                ticktext=[format_tick_label(y) for y in y_ticks],
                                row=row, col=col
                            )
                        idx += 1
                    fig.update_layout(height=500 * n_rows, width=500 * n_cols, showlegend=False)
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select at least 2 elements for scatter plot analysis")

                # Significant Intervals
                st.header("Significant Intervals")
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_length = st.number_input("Minimum Interval Length (m)", value=2.0, min_value=0.1, step=0.5)
                with col2:
                    max_internal_waste = st.number_input("Maximum Internal Waste (m)", value=2.0, min_value=0.0, step=0.5)
                with col3:
                    interval_cutoff = st.number_input(
                        f"Minimum {primary_element} Grade",
                        value=float(st.session_state.merged_df[primary_element].median()),
                        min_value=float(st.session_state.merged_df[primary_element].min()),
                        max_value=float(st.session_state.merged_df[primary_element].max()),
                        step=0.1
                    )
                
                if st.button("Calculate Significant Intervals"):
                    st.session_state.significant_intervals = calculate_significant_intervals(
                        st.session_state.merged_df, primary_element, interval_cutoff, min_length, max_internal_waste, st.session_state.litho_dict
                    )
                    if not st.session_state.significant_intervals.empty:
                        st.write(st.session_state.significant_intervals)
                    else:
                        st.warning("No significant intervals found with current parameters.")
            else:
                st.warning("No data available for analysis after applying filters.")
        
        # Lithology analysis
        if st.session_state.analysis_mode in ["collar_litho", "all"] and 'LITHO' in st.session_state.merged_df.columns:
            if not st.session_state.merged_df.empty:
                if st.session_state.analysis_mode == "collar_litho":
                    # For lithology-only data, we need a dummy element for analysis
                    st.session_state.merged_df['DUMMY'] = 1.0
                    primary_element = 'DUMMY'
                else:
                    primary_element = st.selectbox("Select element for lithology analysis:", 
                                                  st.session_state.element_cols,
                                                  key="litho_primary_element")
                
                use_log_scale = st.checkbox("Use log scale for lithology analysis", value=True, key="litho_log_scale")
                create_lithology_analysis(st.session_state.merged_df, primary_element, use_log_scale, st.session_state.litho_dict)
            else:
                st.warning("No data available for lithology analysis after applying filters.")
    else:
        st.warning("Please load data in the Data Loading tab first.")

with tab_clustering:
    st.header("Geochemical Clustering")
    
    if st.session_state.merged_df is not None and st.session_state.analysis_mode in ["collar_assay", "all"]:
        if 'selected_cluster_features' not in state or state.selected_cluster_features is None:
            state.selected_cluster_features = st.session_state.element_cols[:min(5, len(st.session_state.element_cols))]

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Select All Features"):
                state.selected_cluster_features = st.session_state.element_cols.copy()
        with col2:
            cluster_features = st.multiselect(
                "Select features for clustering",
                st.session_state.element_cols,
                default=state.selected_cluster_features,
                key="cluster_features_select"
            )

        use_pca = st.checkbox("Perform PCA before clustering", value=False)
        use_log_transform = st.checkbox("Apply natural log transform", value=False)
        if use_pca:
            n_components = st.slider("Maximum number of components to consider", 2, len(cluster_features), min(3, len(cluster_features)))
        max_clusters = st.slider("Maximum number of clusters to consider", min_value=2, max_value=15, value=5)

        # Step 2: Confirmation
        if st.button("Confirm Selection and Analyse Clusters"):
            # Update the session state
            state.selected_cluster_features = cluster_features
            
            # Run clustering analysis
            cluster_df = st.session_state.merged_df.copy()
            state.X_scaled, state.scaler, state.wcss = run_clustering_analysis(
                cluster_df, cluster_features, use_pca, use_log_transform, 
                n_components if use_pca else None, max_clusters
            )
            
            if state.X_scaled is not None:
                st.subheader("Clustering Scree Plot")
                st.plotly_chart(plot_scree(wcss=state.wcss, is_pca=False))

                state.n_clusters = st.number_input("Select number of clusters", min_value=2, max_value=max_clusters, value=state.n_clusters)
                kmeans = KMeans(n_clusters=state.n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(state.X_scaled)

                # Add cluster labels to dataframes
                cluster_df['Cluster'] = cluster_labels
                
                # Create a mapping from (HOLE_ID, FROM, TO) to Cluster
                cluster_mapping = dict(zip(
                    zip(cluster_df['HOLE_ID'], cluster_df['FROM'], cluster_df['TO']),
                    cluster_df['Cluster']
                ))
                
                # Apply mapping to merged_df
                st.session_state.merged_df['Cluster'] = [
                    cluster_mapping.get((hole, from_val, to_val), -1)
                    for hole, from_val, to_val in zip(st.session_state.merged_df['HOLE_ID'], st.session_state.merged_df['FROM'], st.session_state.merged_df['TO'])
                ]
                
                # Apply mapping to viz_df
                st.session_state.viz_df['Cluster'] = [
                    cluster_mapping.get((hole, from_val, to_val), -1)
                    for hole, from_val, to_val in zip(st.session_state.viz_df['HOLE_ID'], st.session_state.viz_df['FROM'], st.session_state.viz_df['TO'])
                ]
                
                # Remove any rows that didn't get a valid cluster assignment
                st.session_state.merged_df = st.session_state.merged_df[st.session_state.merged_df['Cluster'] >= 0]
                st.session_state.viz_df = st.session_state.viz_df[st.session_state.viz_df['Cluster'] >= 0]

                if not use_pca:
                    if use_log_transform:
                        cluster_centers = pd.DataFrame(
                            np.exp(state.scaler.inverse_transform(kmeans.cluster_centers_)) - 1e-10,
                            columns=cluster_features
                        )
                    else:
                        cluster_centers = pd.DataFrame(
                            state.scaler.inverse_transform(kmeans.cluster_centers_),
                            columns=cluster_features
                        )
                    st.write("Cluster Centers:")
                    st.write(cluster_centers)

                st.subheader("3D PCA of Clusters")
                def plot_3d_pca(X_scaled, n_clusters, kmeans, feature_names):
                    n_components_here = X_scaled.shape[1]
                    pca_3d = PCA(n_components=min(3, n_components_here))
                    X_pca_3d = pca_3d.fit_transform(X_scaled)
                    fig_3d = go.Figure()
                    if n_components_here >= 3:
                        for i in range(n_clusters):
                            cluster_points = X_pca_3d[kmeans.labels_ == i]
                            fig_3d.add_trace(go.Scatter3d(
                                x=cluster_points[:, 0],
                                y=cluster_points[:, 1],
                                z=cluster_points[:, 2],
                                mode='markers',
                                marker=dict(size=6),
                                name=f'Cluster {i}'
                            ))
                        centroids_pca = pca_3d.transform(kmeans.cluster_centers_)
                        fig_3d.add_trace(go.Scatter3d(
                            x=centroids_pca[:, 0],
                            y=centroids_pca[:, 1],
                            z=centroids_pca[:, 2],
                            mode='markers',
                            marker=dict(color='black', size=10, symbol='diamond'),
                            name='Centroids'
                        ))
                        loadings = pca_3d.components_.T * np.sqrt(pca_3d.explained_variance_)
                        data_range = np.max(np.abs(X_pca_3d))
                        loading_range = np.max(np.abs(loadings))
                        scaling_factor = (data_range / loading_range) * 0.8
                        for i, feature in enumerate(feature_names):
                            fig_3d.add_trace(go.Scatter3d(
                                x=[0, loadings[i, 0] * scaling_factor],
                                y=[0, loadings[i, 1] * scaling_factor],
                                z=[0, loadings[i, 2] * scaling_factor],
                                mode='lines+text',
                                line=dict(color='red', width=3),
                                text=['', feature],
                                textposition='top center',
                                textfont=dict(size=12),
                                name=f'Loading {feature}',
                                showlegend=False
                            ))
                        fig_3d.update_layout(
                            title='3D PCA of Clusters',
                            scene=dict(
                                xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
                                yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
                                zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})',
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                            ),
                            width=1800,  # Increased from 1400 to 1800
                            height=1200  # Increased from 1000 to 1200
                        )
                    else:
                        for i in range(n_clusters):
                            cluster_points = X_pca_3d[kmeans.labels_ == i]
                            fig_3d.add_trace(go.Scatter(
                                x=cluster_points[:, 0],
                                y=cluster_points[:, 1],
                                mode='markers',
                                marker=dict(size=6),
                                name=f'Cluster {i}'
                            ))
                        centroids_pca = pca_3d.transform(kmeans.cluster_centers_)
                        fig_3d.add_trace(go.Scatter(
                            x=centroids_pca[:, 0],
                            y=centroids_pca[:, 1],
                            mode='markers',
                            marker=dict(color='black', size=10, symbol='diamond'),
                            name='Centroids'
                        ))
                        loadings = pca_3d.components_.T * np.sqrt(pca_3d.explained_variance_)
                        data_range = np.max(np.abs(X_pca_3d))
                        loading_range = np.max(np.abs(loadings))
                        scaling_factor = (data_range / loading_range) * 0.8
                        for i, feature in enumerate(feature_names):
                            fig_3d.add_trace(go.Scatter(
                                x=[0, loadings[i, 0] * scaling_factor],
                                y=[0, loadings[i, 1] * scaling_factor],
                                mode='lines+text',
                                line=dict(color='red', width=3),
                                text=['', feature],
                                textposition='top center',
                                textfont=dict(size=12),
                                name=f'Loading {feature}',
                                showlegend=False
                            ))
                        fig_3d.update_layout(
                            title='2D PCA of Clusters',
                            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
                            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
                            width=1800,  # Increased from 1400 to 1800
                            height=1200   # Increased from 1000 to 1200
                        )
                    fig_3d.update_layout(
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.02,
                            font=dict(size=12)
                        ),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    fig_3d.add_annotation(
                        text="Red lines show feature loadings",
                        xref="paper", yref="paper",
                        x=0, y=1.05,
                        showarrow=False,
                        font=dict(size=14)
                    )
                    return fig_3d

                fig_3d_pca = plot_3d_pca(state.X_scaled, state.n_clusters, kmeans, cluster_features if not use_pca else [f"PC{i+1}" for i in range(n_components)])
                st.plotly_chart(fig_3d_pca)

                st.subheader("Cluster Summary Statistics")
                primary_element_for_cluster = st.selectbox("Select element for cluster comparison:", st.session_state.element_cols)
                summary_stats = get_cluster_summary(st.session_state.merged_df, cluster_features, primary_element_for_cluster)
                st.write(summary_stats.round(3))

                st.subheader("Feature Distribution by Cluster")
                use_log_scale_cluster = st.checkbox("Use log scale for boxplots", value=True)
                fig_boxplots = plot_cluster_boxplots(st.session_state.merged_df, cluster_features, primary_element_for_cluster, use_log_scale_cluster)
                st.plotly_chart(fig_boxplots)

                if 'LITHO' in st.session_state.merged_df.columns:
                    st.subheader("Lithology vs Cluster Comparison")
                    fig_lith_cluster = plot_lithology_cluster_comparison(st.session_state.merged_df)
                    st.plotly_chart(fig_lith_cluster)

                # Cluster visualisation
                st.subheader("3D Visualisation of Clusters")

                if 'Cluster' in st.session_state.merged_df.columns:
                    # Create a copy of the visualisation dataframe
                    cluster_viz_df = st.session_state.viz_df.copy()
                    
                    # Filter to only include rows with valid cluster assignments
                    cluster_viz_df = cluster_viz_df[cluster_viz_df['Cluster'] >= 0]
                    
                    if not cluster_viz_df.empty:
                        # Create visualisation options
                        viz_options = ["Clusters"]
                        if 'primary_element_for_cluster' in locals():
                            viz_options.append("Grade")
                        if 'LITHO' in cluster_viz_df.columns:
                            viz_options.append("Lithology")
                            
                        # Use multiselect instead of dropdown
                        viz_type = st.multiselect(
                        "Select visualisation types:",
                        viz_options,
                        default=["Clusters"],
                        key="cluster_viz_type_selection"  
                    )
                        
                        vertical_exaggeration = st.slider(
                            "Vertical Exaggeration", 
                            min_value=1.0, 
                            max_value=10.0, 
                            value=1.0, 
                            step=0.1,
                            key="cluster_viz_exaggeration_slider"
                        )
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # Get collar dataframe for visualisation
                        viz_collar_df = st.session_state.collar_df.copy()
                        
                        # Define offsets for different visualisation types
                        offsets = {"Clusters": 0, "Grade": 20, "Lithology": -20}
                        
                        # Add selected visualisation types
                        for v_type in viz_type:
                            if v_type == "Clusters":
                                # Add cluster visualisation
                                for cluster in sorted(cluster_viz_df['Cluster'].unique()):
                                    if cluster >= 0:  # Skip invalid clusters
                                        cluster_data = cluster_viz_df[cluster_viz_df['Cluster'] == cluster]
                                        if not cluster_data.empty:
                                            hover_text = []
                                            for _, row in cluster_data.iterrows():
                                                info = [
                                                    f"<b>Hole ID:</b> {row['HOLE_ID']}",
                                                    f"<b>Cluster:</b> {cluster}"
                                                ]
                                                if 'primary_element_for_cluster' in locals():
                                                    info.append(f"<b>{primary_element_for_cluster}:</b> {row[primary_element_for_cluster]:.2f}")
                                                info.append(f"<b>From:</b> {row['FROM']:.2f}")
                                                info.append(f"<b>To:</b> {row['TO']:.2f}")
                                                if 'LITHO' in row:
                                                    info.append(f"<b>Lithology:</b> {row['LITHO']}")
                                                hover_text.append("<br>".join(info))
                                            
                                            fig.add_trace(go.Scatter3d(
                                                x=cluster_data['x'] + offsets["Clusters"],
                                                y=cluster_data['y'],
                                                z=cluster_data['z'],
                                                mode='markers',
                                                marker=dict(
                                                    size=8,  # Increased marker size from 5 to 8
                                                    color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]
                                                ),
                                                name=f'Cluster {cluster}',
                                                hovertemplate="%{text}<br>" +
                                                            "<b>X:</b> %{x:.2f}<br>" +
                                                            "<b>Y:</b> %{y:.2f}<br>" +
                                                            "<b>Z:</b> %{z:.2f}<extra></extra>",
                                                text=hover_text
                                            ))
                                
                                # Add drill hole lines for clusters
                                for hole in cluster_viz_df['HOLE_ID'].unique():
                                    hole_data = cluster_viz_df[cluster_viz_df['HOLE_ID'] == hole]
                                    collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                                    if not collar_point.empty:
                                        collar_point = collar_point.iloc[0]
                                        x_line = [collar_point['EASTING'] + offsets["Clusters"]] + (hole_data['x'] + offsets["Clusters"]).tolist()
                                        y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                                        z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                                        fig.add_trace(go.Scatter3d(
                                            x=x_line, y=y_line, z=z_line,
                                            mode='lines',
                                            line=dict(color='gray', width=1),
                                            showlegend=False
                                        ))
                                
                                # Add collar points for clusters
                                add_collar_points(fig, viz_collar_df, x_offset=offsets["Clusters"])
                            
                            elif v_type == "Grade" and 'primary_element_for_cluster' in locals():
                                # Add grade visualisation
                                use_log_scale = st.checkbox("Use log scale for grade", value=True, key="cluster_grade_log")
                                add_grade_visualisation(fig, cluster_viz_df, primary_element_for_cluster, use_log_scale, "Combined", x_offset=offsets["Grade"])
                                
                                # Add drill hole lines for grade
                                for hole in cluster_viz_df['HOLE_ID'].unique():
                                    hole_data = cluster_viz_df[cluster_viz_df['HOLE_ID'] == hole]
                                    collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                                    if not collar_point.empty:
                                        collar_point = collar_point.iloc[0]
                                        x_line = [collar_point['EASTING'] + offsets["Grade"]] + (hole_data['x'] + offsets["Grade"]).tolist()
                                        y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                                        z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                                        fig.add_trace(go.Scatter3d(
                                            x=x_line, y=y_line, z=z_line,
                                            mode='lines',
                                            line=dict(color='gray', width=1),
                                            showlegend=False
                                        ))
                                
                                # Add collar points for grade
                                add_collar_points(fig, viz_collar_df, x_offset=offsets["Grade"])
                            
                            elif v_type == "Lithology" and 'LITHO' in cluster_viz_df.columns:
                                # Create a lithology visualisation dataframe
                                litho_viz_df = cluster_viz_df.copy()
                                add_lithology_visualisation(fig, litho_viz_df, "Combined", None, st.session_state.litho_dict, x_offset=offsets["Lithology"])
                                
                                # Add drill hole lines for lithology
                                for hole in litho_viz_df['HOLE_ID'].unique():
                                    hole_data = litho_viz_df[litho_viz_df['HOLE_ID'] == hole]
                                    collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                                    if not collar_point.empty:
                                        collar_point = collar_point.iloc[0]
                                        x_line = [collar_point['EASTING'] + offsets["Lithology"]] + (hole_data['x'] + offsets["Lithology"]).tolist()
                                        y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                                        z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                                        fig.add_trace(go.Scatter3d(
                                            x=x_line, y=y_line, z=z_line,
                                            mode='lines',
                                            line=dict(color='gray', width=1),
                                            showlegend=False
                                        ))
                                
                                # Add collar points for lithology
                                add_collar_points(fig, viz_collar_df, x_offset=offsets["Lithology"])
                        
                        # Update layout
                        update_figure_layout(fig, vertical_exaggeration)
                        
                        # Display the figure
                        st.plotly_chart(fig)
                    else:
                        st.warning("No cluster data available for visualisation after applying filters.")
                else:
                    st.warning("No cluster data available. Please run clustering analysis first.")
                    
                                                                    
                if st.session_state.significant_intervals is not None:
                    st.session_state.significant_intervals = pd.merge(
                        st.session_state.significant_intervals,
                        st.session_state.merged_df[['HOLE_ID', 'FROM', 'Cluster']],
                        on=['HOLE_ID', 'FROM'],
                        how='left'
                    )
                    st.write("Significant Intervals with Cluster Information:")
                    st.write(st.session_state.significant_intervals)

        # Display previous results if available
        elif state.X_scaled is not None:

            st.subheader("Clustering Scree Plot")
            st.plotly_chart(plot_scree(wcss=state.wcss, is_pca=False))
            
            state.n_clusters = st.number_input("Select number of clusters", min_value=2, max_value=max_clusters, value=state.n_clusters)
            kmeans = KMeans(n_clusters=state.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(state.X_scaled)
            
            # Add cluster labels to dataframes
            cluster_df = st.session_state.merged_df.copy()
            cluster_df['Cluster'] = cluster_labels
            
            # Create a mapping from (HOLE_ID, FROM, TO) to Cluster
            cluster_mapping = dict(zip(
                zip(cluster_df['HOLE_ID'], cluster_df['FROM'], cluster_df['TO']),
                cluster_df['Cluster']
            ))
            
            # Apply mapping to merged_df
            st.session_state.merged_df['Cluster'] = [
                cluster_mapping.get((hole, from_val, to_val), -1)
                for hole, from_val, to_val in zip(st.session_state.merged_df['HOLE_ID'], st.session_state.merged_df['FROM'], st.session_state.merged_df['TO'])
            ]
            
            # Apply mapping to viz_df
            st.session_state.viz_df['Cluster'] = [
                cluster_mapping.get((hole, from_val, to_val), -1)
                for hole, from_val, to_val in zip(st.session_state.viz_df['HOLE_ID'], st.session_state.viz_df['FROM'], st.session_state.viz_df['TO'])
            ]
            
            # Remove any rows that didn't get a valid cluster assignment
            st.session_state.merged_df = st.session_state.merged_df[st.session_state.merged_df['Cluster'] >= 0]
            st.session_state.viz_df = st.session_state.viz_df[st.session_state.viz_df['Cluster'] >= 0]
            
            if not use_pca:
                if use_log_transform:
                    cluster_centers = pd.DataFrame(
                        np.exp(state.scaler.inverse_transform(kmeans.cluster_centers_)) - 1e-10,
                        columns=cluster_features
                    )
                else:
                    cluster_centers = pd.DataFrame(
                        state.scaler.inverse_transform(kmeans.cluster_centers_),
                        columns=cluster_features
                    )
                st.write("Cluster Centers:")
                st.write(cluster_centers)

            st.subheader("3D PCA of Clusters")
            def plot_3d_pca(X_scaled, n_clusters, kmeans, feature_names):
                n_components_here = X_scaled.shape[1]
                pca_3d = PCA(n_components=min(3, n_components_here))
                X_pca_3d = pca_3d.fit_transform(X_scaled)
                fig_3d = go.Figure()
                if n_components_here >= 3:
                    for i in range(n_clusters):
                        cluster_points = X_pca_3d[kmeans.labels_ == i]
                        fig_3d.add_trace(go.Scatter3d(
                            x=cluster_points[:, 0],
                            y=cluster_points[:, 1],
                            z=cluster_points[:, 2],
                            mode='markers',
                            marker=dict(size=6),
                            name=f'Cluster {i}'
                        ))
                    centroids_pca = pca_3d.transform(kmeans.cluster_centers_)
                    fig_3d.add_trace(go.Scatter3d(
                        x=centroids_pca[:, 0],
                        y=centroids_pca[:, 1],
                        z=centroids_pca[:, 2],
                        mode='markers',
                        marker=dict(color='black', size=10, symbol='diamond'),
                        name='Centroids'
                    ))
                    loadings = pca_3d.components_.T * np.sqrt(pca_3d.explained_variance_)
                    data_range = np.max(np.abs(X_pca_3d))
                    loading_range = np.max(np.abs(loadings))
                    scaling_factor = (data_range / loading_range) * 0.8
                    for i, feature in enumerate(feature_names):
                        fig_3d.add_trace(go.Scatter3d(
                            x=[0, loadings[i, 0] * scaling_factor],
                            y=[0, loadings[i, 1] * scaling_factor],
                            z=[0, loadings[i, 2] * scaling_factor],
                            mode='lines+text',
                            line=dict(color='red', width=3),
                            text=['', feature],
                            textposition='top center',
                            textfont=dict(size=12),
                            name=f'Loading {feature}',
                            showlegend=False
                        ))
                    fig_3d.update_layout(
                        title='3D PCA of Clusters',
                        scene=dict(
                            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
                            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
                            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        width=1800,  # Increased from 1400 to 1800
                        height=1200  # Increased from 1000 to 1200
                    )
                else:
                    for i in range(n_clusters):
                        cluster_points = X_pca_3d[kmeans.labels_ == i]
                        fig_3d.add_trace(go.Scatter(
                            x=cluster_points[:, 0],
                            y=cluster_points[:, 1],
                            mode='markers',
                            marker=dict(size=6),
                            name=f'Cluster {i}'
                        ))
                    centroids_pca = pca_3d.transform(kmeans.cluster_centers_)
                    fig_3d.add_trace(go.Scatter(
                        x=centroids_pca[:, 0],
                        y=centroids_pca[:, 1],
                        mode='markers',
                        marker=dict(color='black', size=10, symbol='diamond'),
                        name='Centroids'
                    ))
                    loadings = pca_3d.components_.T * np.sqrt(pca_3d.explained_variance_)
                    data_range = np.max(np.abs(X_pca_3d))
                    loading_range = np.max(np.abs(loadings))
                    scaling_factor = (data_range / loading_range) * 0.8
                    for i, feature in enumerate(feature_names):
                        fig_3d.add_trace(go.Scatter(
                            x=[0, loadings[i, 0] * scaling_factor],
                            y=[0, loadings[i, 1] * scaling_factor],
                            mode='lines+text',
                            line=dict(color='red', width=3),
                            text=['', feature],
                            textposition='top center',
                            textfont=dict(size=12),
                            name=f'Loading {feature}',
                            showlegend=False
                        ))
                    fig_3d.update_layout(
                        title='2D PCA of Clusters',
                        xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
                        yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
                        width=1800,  # Increased from 1400 to 1800
                        height=1200  # Increased from 1000 to 1200
                    )
                fig_3d.update_layout(
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=12)
                    ),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                fig_3d.add_annotation(
                    text="Red lines show feature loadings",
                    xref="paper", yref="paper",
                    x=0, y=1.05,
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig_3d

            fig_3d_pca = plot_3d_pca(state.X_scaled, state.n_clusters, kmeans, cluster_features if not use_pca else [f"PC{i+1}" for i in range(n_components)])
            st.plotly_chart(fig_3d_pca)

            st.subheader("Cluster Summary Statistics")
            primary_element_for_cluster = st.selectbox("Select element for cluster comparison:", st.session_state.element_cols)
            summary_stats = get_cluster_summary(st.session_state.merged_df, cluster_features, primary_element_for_cluster)
            st.write(summary_stats.round(3))

            st.subheader("Feature Distribution by Cluster")
            use_log_scale_cluster = st.checkbox("Use log scale for boxplots", value=True)
            fig_boxplots = plot_cluster_boxplots(st.session_state.merged_df, cluster_features, primary_element_for_cluster, use_log_scale_cluster)
            st.plotly_chart(fig_boxplots)

            if 'LITHO' in st.session_state.merged_df.columns:
                st.subheader("Lithology vs Cluster Comparison")
                fig_lith_cluster = plot_lithology_cluster_comparison(st.session_state.merged_df)
                st.plotly_chart(fig_lith_cluster)

            # Cluster visualisation
            st.subheader("3D Visualisation of Clusters")

            if 'Cluster' in st.session_state.merged_df.columns:
                # Create a copy of the visualisation dataframe
                cluster_viz_df = st.session_state.viz_df.copy()
                
                # Filter to only include rows with valid cluster assignments
                cluster_viz_df = cluster_viz_df[cluster_viz_df['Cluster'] >= 0]
                
                if not cluster_viz_df.empty:
                    # Create visualisation options
                    viz_options = ["Clusters"]
                    if 'primary_element_for_cluster' in locals():
                        viz_options.append("Grade")
                    if 'LITHO' in cluster_viz_df.columns:
                        viz_options.append("Lithology")
                        
                    # Use multiselect with a unique key
                    viz_type = st.multiselect(
                        "Select visualisation types:",
                        viz_options,
                        default=["Clusters"],
                        key="previous_cluster_viz_type_selection"  # Different key for this section
                    )

                    

                    vertical_exaggeration = st.slider(
                        "Vertical Exaggeration", 
                        min_value=1.0, 
                        max_value=10.0, 
                        value=1.0, 
                        step=0.1,
                        key="cluster_viz_exaggeration_previous"  # Keep the existing key
                    )

                    # Create figure
                    fig = go.Figure()

                    # Get collar dataframe for visualisation
                    viz_collar_df = st.session_state.collar_df.copy()

                    # Define offsets for different visualisation types
                    offsets = {"Clusters": 0, "Grade": 20, "Lithology": -20}

                    # Add selected visualisation types
                    for v_type in viz_type:
                        if v_type == "Clusters":
                            # Add cluster visualisation
                            for cluster in sorted(cluster_viz_df['Cluster'].unique()):
                                if cluster >= 0:  # Skip invalid clusters
                                    cluster_data = cluster_viz_df[cluster_viz_df['Cluster'] == cluster]
                                    if not cluster_data.empty:
                                        hover_text = []
                                        for _, row in cluster_data.iterrows():
                                            info = [
                                                f"<b>Hole ID:</b> {row['HOLE_ID']}",
                                                f"<b>Cluster:</b> {cluster}"
                                            ]
                                            if 'primary_element_for_cluster' in locals():
                                                info.append(f"<b>{primary_element_for_cluster}:</b> {row[primary_element_for_cluster]:.2f}")
                                            info.append(f"<b>From:</b> {row['FROM']:.2f}")
                                            info.append(f"<b>To:</b> {row['TO']:.2f}")
                                            if 'LITHO' in row:
                                                info.append(f"<b>Lithology:</b> {row['LITHO']}")
                                            hover_text.append("<br>".join(info))
                                        
                                        fig.add_trace(go.Scatter3d(
                                            x=cluster_data['x'] + offsets["Clusters"],
                                            y=cluster_data['y'],
                                            z=cluster_data['z'],
                                            mode='markers',
                                            marker=dict(
                                                size=8,  # Increased marker size from 5 to 8
                                                color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]
                                            ),
                                            name=f'Cluster {cluster}',
                                            hovertemplate="%{text}<br>" +
                                                        "<b>X:</b> %{x:.2f}<br>" +
                                                        "<b>Y:</b> %{y:.2f}<br>" +
                                                        "<b>Z:</b> %{z:.2f}<extra></extra>",
                                            text=hover_text
                                        ))
                            
                            # Add drill hole lines for clusters
                            for hole in cluster_viz_df['HOLE_ID'].unique():
                                hole_data = cluster_viz_df[cluster_viz_df['HOLE_ID'] == hole]
                                collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                                if not collar_point.empty:
                                    collar_point = collar_point.iloc[0]
                                    x_line = [collar_point['EASTING'] + offsets["Clusters"]] + (hole_data['x'] + offsets["Clusters"]).tolist()
                                    y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                                    z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                                    fig.add_trace(go.Scatter3d(
                                        x=x_line, y=y_line, z=z_line,
                                        mode='lines',
                                        line=dict(color='gray', width=1),
                                        showlegend=False
                                    ))
                            
                            # Add collar points for clusters
                            add_collar_points(fig, viz_collar_df, x_offset=offsets["Clusters"])
                        
                        elif v_type == "Grade" and 'primary_element_for_cluster' in locals():
                            # Add grade visualisation
                            use_log_scale = st.checkbox("Use log scale for grade", value=True, key="previous_cluster_grade_log")  # Different key
                            add_grade_visualisation(fig, cluster_viz_df, primary_element_for_cluster, use_log_scale, "Combined", color_by='grade', x_offset=offsets["Grade"])
                            
                            # Add drill hole lines for grade
                            for hole in cluster_viz_df['HOLE_ID'].unique():
                                hole_data = cluster_viz_df[cluster_viz_df['HOLE_ID'] == hole]
                                collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                                if not collar_point.empty:
                                    collar_point = collar_point.iloc[0]
                                    x_line = [collar_point['EASTING'] + offsets["Grade"]] + (hole_data['x'] + offsets["Grade"]).tolist()
                                    y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                                    z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                                    fig.add_trace(go.Scatter3d(
                                        x=x_line, y=y_line, z=z_line,
                                        mode='lines',
                                        line=dict(color='gray', width=1),
                                        showlegend=False
                                    ))
                            
                            # Add collar points for grade
                            add_collar_points(fig, viz_collar_df, x_offset=offsets["Grade"])
                        
                        elif v_type == "Lithology" and 'LITHO' in cluster_viz_df.columns:
                            # Add lithology visualisation
                            add_lithology_visualisation(fig, cluster_viz_df, "Combined", None, st.session_state.litho_dict, x_offset=offsets["Lithology"])
                            
                            # Add drill hole lines for lithology
                            for hole in cluster_viz_df['HOLE_ID'].unique():
                                hole_data = cluster_viz_df[cluster_viz_df['HOLE_ID'] == hole]
                                collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                                if not collar_point.empty:
                                    collar_point = collar_point.iloc[0]
                                    x_line = [collar_point['EASTING'] + offsets["Lithology"]] + (hole_data['x'] + offsets["Lithology"]).tolist()
                                    y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                                    z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                                    fig.add_trace(go.Scatter3d(
                                        x=x_line, y=y_line, z=z_line,
                                        mode='lines',
                                        line=dict(color='gray', width=1),
                                        showlegend=False
                                    ))
                            
                            # Add collar points for lithology
                            add_collar_points(fig, viz_collar_df, x_offset=offsets["Lithology"])
                            if 'primary_element_for_cluster' in locals():
                                info.append(f"<b>{primary_element_for_cluster}:</b> {row[primary_element_for_cluster]:.2f}")
                            info.append(f"<b>From:</b> {row['FROM']:.2f}")
                            info.append(f"<b>To:</b> {row['TO']:.2f}")
                            if 'LITHO' in row:
                                info.append(f"<b>Lithology:</b> {row['LITHO']}")
                            hover_text.append("<br>".join(info))
                        
                        fig.add_trace(go.Scatter3d(
                            x=cluster_data['x'],
                            y=cluster_data['y'],
                            z=cluster_data['z'],
                            mode='markers',
                            marker=dict(
                                size=8,  # Increased marker size from 5 to 8
                                color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]
                            ),
                            name=f'Cluster {cluster}',
                            hovertemplate="%{text}<br>" +
                                        "<b>X:</b> %{x:.2f}<br>" +
                                        "<b>Y:</b> %{y:.2f}<br>" +
                                        "<b>Z:</b> %{z:.2f}<extra></extra>",
                            text=hover_text
                        ))        
            # Add drill hole lines
            for hole in cluster_viz_df['HOLE_ID'].unique():
                hole_data = cluster_viz_df[cluster_viz_df['HOLE_ID'] == hole]
                collar_point = viz_collar_df[viz_collar_df['HOLE_ID'] == hole]
                if not collar_point.empty:
                    collar_point = collar_point.iloc[0]
                    x_line = [collar_point['EASTING']] + hole_data['x'].tolist()
                    y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                    z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                    fig.add_trace(go.Scatter3d(
                        x=x_line, y=y_line, z=z_line,
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))
            
            # Add collar points
            add_collar_points(fig, viz_collar_df)
            
            # Update layout
            update_figure_layout(fig, vertical_exaggeration)
            
            st.plotly_chart(fig, key="previous_cluster_viz_plot")  # Different key here
        else:
            st.warning("No cluster data available for visualisation after applying filters.")
    else:
        st.warning("No cluster data available. Please run clustering analysis first.")
        
with tab_download:
    st.header("Download Data")
    
    if st.session_state.merged_df is not None:
        download_cols = st.columns(4)
        with download_cols[0]:
            csv_processed = st.session_state.merged_df.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv_processed,
                file_name="processed_drillhole_data.csv",
                mime="text/csv"
            )

        if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
            with download_cols[1]:
                primary_element_for_stats = st.session_state.element_cols[0] if len(st.session_state.element_cols) > 0 else None
                if primary_element_for_stats and primary_element_for_stats in st.session_state.merged_df.columns:
                    stats_dict = {
                        'Statistic': [
                            'Count', 'Mean', 'Median', 'Std Dev', 'CV', 
                            'Min', 'Q1', 'Q3', 'Max', 'Skewness', 'Kurtosis'
                        ],
                        'Value': [
                            len(st.session_state.merged_df[primary_element_for_stats]),
                            st.session_state.merged_df[primary_element_for_stats].mean(),
                            st.session_state.merged_df[primary_element_for_stats].median(),
                            st.session_state.merged_df[primary_element_for_stats].std(),
                            st.session_state.merged_df[primary_element_for_stats].std() / st.session_state.merged_df[primary_element_for_stats].mean(),
                            st.session_state.merged_df[primary_element_for_stats].min(),
                            st.session_state.merged_df[primary_element_for_stats].quantile(0.25),
                            st.session_state.merged_df[primary_element_for_stats].quantile(0.75),
                            st.session_state.merged_df[primary_element_for_stats].max(),
                            st.session_state.merged_df[primary_element_for_stats].skew(),
                            st.session_state.merged_df[primary_element_for_stats].kurtosis()
                        ]
                    }
                    stats_df = pd.DataFrame(stats_dict)
                    stats_df['Value'] = stats_df['Value'].round(3)
                    csv_stats = stats_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {primary_element_for_stats} Statistics",
                        data=csv_stats,
                        file_name=f"{primary_element_for_stats}_statistics.csv",
                        mime="text/csv"
                    )

        if st.session_state.significant_intervals is not None and not st.session_state.significant_intervals.empty:
            with download_cols[2]:
                csv_intervals = st.session_state.significant_intervals.to_csv(index=False)
                st.download_button(
                    label="Download Significant Intervals",
                    data=csv_intervals,
                    file_name="significant_intervals.csv",
                    mime="text/csv"
                )

        if 'LITHO' in st.session_state.merged_df.columns:
            with download_cols[3]:
                litho_stats = st.session_state.merged_df.groupby('LITHO').size().reset_index(name='Count')
                if st.session_state.litho_dict:
                    litho_stats['Description'] = litho_stats['LITHO'].map(lambda x: st.session_state.litho_dict.get(x, ""))
                csv_litho_stats = litho_stats.to_csv(index=False)
                st.download_button(
                    label="Download Lithology Statistics",
                    data=csv_litho_stats,
                    file_name="lithology_statistics.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Please load data in the Data Loading tab first.")
