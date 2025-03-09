import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="3DA",
    page_icon="🔍",
    layout="wide"
)
def read_mining_csv(file):
    """
    Read CSV files with mining software format where:
    - Headers are in a row starting with H1000
    - Data rows start with D
    """
    try:
        # Read the  file first
        df = pd.read_csv(file)
        
        # Find the header row (H1000)
        header_row_idx = df[df.iloc[:, 0] == 'H1000'].index[0]
        headers = df.iloc[header_row_idx].values.tolist()
        
        # Find all data rows (starting with 'D')
        data_rows = df[df.iloc[:, 0] == 'D']
        
        # Create new dataframe with correct headers and data
        result_df = pd.DataFrame(data_rows.values, columns=headers)
        
        # Remove the 'D' column (first column)
        result_df = result_df.iloc[:, 1:]
        
        return result_df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def create_swath_plots(merged_df, primary_element, use_log_scale):
    """Create swath plots for Easting, Northing, and Elevation"""
    st.header("Swath Plots")
    
    # Create tabs for different swath directions
    tab1, tab2, tab3 = st.tabs(["Easting Swath", "Northing Swath", "Elevation Swath"])

    def create_swath_data(df, coord_col, value_col, num_bins=20):
        bins = np.linspace(df[coord_col].min(), df[coord_col].max(), num_bins + 1)
        df['bin'] = pd.cut(df[coord_col], bins)
        swath_stats = df.groupby('bin')[value_col].agg(['mean', 'count', 'std']).reset_index()
        swath_stats['bin_center'] = [(x.left + x.right)/2 for x in swath_stats['bin']]
        return swath_stats

    with tab1:
        # Easting Swath
        st.subheader("Grade Distribution by Easting")
        easting_swath = create_swath_data(merged_df, 'x', primary_element)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=easting_swath['bin_center'],
            y=easting_swath['mean'],
            mode='lines+markers',
            name='Mean Grade',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Bar(
            x=easting_swath['bin_center'],
            y=easting_swath['count'],
            name='Number of Samples',
            yaxis='y2',
            opacity=0.3
        ))
        fig.add_trace(go.Scatter(
            x=easting_swath['bin_center'],
            y=easting_swath['mean'] + easting_swath['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=easting_swath['bin_center'],
            y=easting_swath['mean'] - easting_swath['std'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,0,255,0.2)',
            fill='tonexty',
            name='± 1 Std Dev'
        ))
        
        fig.update_layout(
            xaxis_title="Easting",
            yaxis_title=primary_element,
            yaxis2=dict(
                title="Number of Samples",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        st.plotly_chart(fig)

    with tab2:
        # Northing Swath
        st.subheader("Grade Distribution by Northing")
        northing_swath = create_swath_data(merged_df, 'y', primary_element)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=northing_swath['bin_center'],
            y=northing_swath['mean'],
            mode='lines+markers',
            name='Mean Grade',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Bar(
            x=northing_swath['bin_center'],
            y=northing_swath['count'],
            name='Number of Samples',
            yaxis='y2',
            opacity=0.3
        ))
        fig.add_trace(go.Scatter(
            x=northing_swath['bin_center'],
            y=northing_swath['mean'] + northing_swath['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=northing_swath['bin_center'],
            y=northing_swath['mean'] - northing_swath['std'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,0,255,0.2)',
            fill='tonexty',
            name='± 1 Std Dev'
        ))
        
        fig.update_layout(
            xaxis_title="Northing",
            yaxis_title=primary_element,
            yaxis2=dict(
                title="Number of Samples",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        st.plotly_chart(fig)

    with tab3:
        # Elevation Swath
        st.subheader("Grade Distribution by Elevation")
        elevation_swath = create_swath_data(merged_df, 'z', primary_element, num_bins=15)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elevation_swath['bin_center'],
            y=elevation_swath['mean'],
            mode='lines+markers',
            name='Mean Grade',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Bar(
            x=elevation_swath['bin_center'],
            y=elevation_swath['count'],
            name='Number of Samples',
            yaxis='y2',
            opacity=0.3
        ))
        fig.add_trace(go.Scatter(
            x=elevation_swath['bin_center'],
            y=elevation_swath['mean'] + elevation_swath['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=elevation_swath['bin_center'],
            y=elevation_swath['mean'] - elevation_swath['std'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,0,255,0.2)',
            fill='tonexty',
            name='± 1 Std Dev'
        ))
        
        fig.update_layout(
            xaxis_title="Elevation",
            yaxis_title=primary_element,
            yaxis2=dict(
                title="Number of Samples",
                overlaying='y',
                side='right'
            ),
            height=400,
            xaxis={'autorange': 'reversed'}
        )
        st.plotly_chart(fig)

def create_lithology_analysis(merged_df, primary_element, use_log_scale):
    """Create lithology analysis plots and statistics"""
    st.header("Lithology Analysis")
    
    # Summary statistics by lithology
    st.subheader(f"Summary Statistics by Lithology - {primary_element}")
    
    # Calculate statistics for each lithology
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
            lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan  # CV
        ]
    })
    
    # Flatten column names and rename
    litho_stats.columns = [
        'Count', 'Mean', 'Median', 'Std Dev',
        'Min', 'Q1', 'Q3', 'Max', 'CV'
    ]
    
    # Round statistics
    litho_stats = litho_stats.round(3)
    
    # Sort by count by default
    litho_stats = litho_stats.sort_values('Count', ascending=False)
    
    # Display statistics
    st.dataframe(litho_stats)
    
    # Create box plots
    st.subheader(f"Grade Distribution by Lithology - {primary_element}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sort options
        sort_by = st.selectbox(
            "Sort Lithologies by:",
            ['Median', 'Mean', 'Count', 'Alphabetical']
        )
        
    with col2:
        # Minimum samples filter
        min_samples = st.number_input(
            "Minimum samples per lithology:",
            min_value=1,
            value=2,
            step=1
        )
    
    # Filter lithologies by minimum sample count
    valid_lithos = litho_stats[litho_stats['Count'] >= min_samples].index
    plot_df = merged_df[merged_df['LITHO'].isin(valid_lithos)].copy()
    
    if not plot_df.empty:
        # Sort lithologies based on selection
        if sort_by == 'Median':
            litho_order = litho_stats.loc[valid_lithos].sort_values('Median', ascending=False).index
        elif sort_by == 'Mean':
            litho_order = litho_stats.loc[valid_lithos].sort_values('Mean', ascending=False).index
        elif sort_by == 'Count':
            litho_order = litho_stats.loc[valid_lithos].sort_values('Count', ascending=False).index
        else:  # Alphabetical
            litho_order = sorted(valid_lithos)
        
        # Create box plot
        fig = go.Figure()
        
        for litho in litho_order:
            litho_data = plot_df[plot_df['LITHO'] == litho][primary_element]
            
            fig.add_trace(go.Box(
                y=litho_data,
                name=litho,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))

        # Calculate y-axis range and ticks
        y_min = plot_df[primary_element].min()
        y_max = plot_df[primary_element].max()
        
        if use_log_scale and y_min > 0:
            # Log scale configuration
            fig.update_layout(
                title=f"{primary_element} Distribution by Lithology",
                yaxis=dict(
                    title=f"{primary_element} (log scale)",
                    type='log',
                    autorange=True,
                    showgrid=True,
                    tickmode='array',
                    tickvals=[10**i for i in range(int(np.floor(np.log10(y_min))), 
                                                 int(np.ceil(np.log10(y_max))) + 1)],
                    ticktext=[f'{10**i:.1f}' for i in range(int(np.floor(np.log10(y_min))), 
                                                          int(np.ceil(np.log10(y_max))) + 1)]
                ),
                showlegend=True,
                height=600,
                boxmode='group'
            )
        else:
            # Linear scale configuration
            tick_interval = (y_max - y_min) / 10  # Create about 10 ticks
            fig.update_layout(
                title=f"{primary_element} Distribution by Lithology",
                yaxis=dict(
                    title=primary_element,
                    type='linear',
                    autorange=True,
                    showgrid=True,
                    dtick=tick_interval,  # Set tick interval
                    tickformat='.2f'  # Format to 2 decimal places
                ),
                showlegend=True,
                height=600,
                boxmode='group'
            )        
        st.plotly_chart(fig)

def add_grade_visualisation(fig, merged_df, primary_element, use_log_scale, viz_mode):
    """Add grade-based visualisation to the figure"""
    # Calculate color values for samples
    if use_log_scale:
        valid_samples = merged_df[merged_df[primary_element] > 0]
        color_values = np.log10(valid_samples[primary_element])
        color_bar_title = f"Log10({primary_element})"
        
        min_log = np.floor(np.log10(valid_samples[primary_element].min()))
        max_log = np.ceil(np.log10(valid_samples[primary_element].max()))
        tick_vals = np.arange(min_log, max_log + 1)
        tick_text = [f'{10**val:.3g}' for val in tick_vals]
    else:
        valid_samples = merged_df
        color_values = valid_samples[primary_element]
        color_bar_title = primary_element
        tick_vals = None
        tick_text = None

    x_offset = -5 if viz_mode == "Combined" else 0
    
    fig.add_trace(go.Scatter3d(
        x=valid_samples['x'] + x_offset,
        y=valid_samples['y'],
        z=valid_samples['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=color_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=color_bar_title,
                ticktext=tick_text,
                tickvals=tick_vals,
            )
        ),
        text=valid_samples[primary_element],
        hovertemplate=(
            f"<b>{primary_element}</b>: %{{text:.3g}}<br>" +
            "<b>Lithology:</b> %{customdata}<br>" +
            "<b>X</b>: %{x:.2f}<br>" +
            "<b>Y</b>: %{y:.2f}<br>" +
            "<b>Z</b>: %{z:.2f}<br>"
        ),
        customdata=valid_samples['LITHO'] if 'LITHO' in valid_samples.columns else [""] * len(valid_samples),
        name='Samples'
    ))
def add_lithology_visualisation(fig, merged_df, viz_mode):
    """Add lithology-based visualisation to the figure"""
    unique_lithos = merged_df['LITHO'].unique()
    litho_colors = px.colors.qualitative.Set3[:len(unique_lithos)]
    litho_color_map = dict(zip(unique_lithos, litho_colors))
    
    x_offset = 5 if viz_mode == "Combined" else 0
    
    for litho in unique_lithos:
        litho_data = merged_df[merged_df['LITHO'] == litho]
        fig.add_trace(go.Scatter3d(
            x=litho_data['x'] + x_offset,
            y=litho_data['y'],
            z=litho_data['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=litho_color_map[litho]
            ),
            name=litho,
            hovertemplate=(
                "<b>Lithology:</b> " + litho + "<br>" +
                "<b>X</b>: %{x:.2f}<br>" +
                "<b>Y</b>: %{y:.2f}<br>" +
                "<b>Z</b>: %{z:.2f}<br>"
            )
        ))

def add_collar_points(fig, collar_df):
    """Add collar points to the figure"""
    fig.add_trace(go.Scatter3d(
        x=collar_df['EASTING'],
        y=collar_df['NORTHING'],
        z=collar_df['ELEVATION'],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='diamond'
        ),
        name='Collar Points'
    ))

def add_drill_traces(fig, merged_df, collar_df, viz_mode):
    """Add drill traces to the figure"""
    for hole in merged_df['HOLE_ID'].unique():
        hole_data = merged_df[merged_df['HOLE_ID'] == hole].sort_values('MIDPOINT')
        collar_point = collar_df[collar_df['HOLE_ID'] == hole].iloc[0]
        
        if viz_mode == "Combined":
            # Add lines for both grade and lithology views
            for offset in [-5, 5]:
                x_line = [collar_point['EASTING'] + offset] + (hole_data['x'] + offset).tolist()
                y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
        else:
            x_line = [collar_point['EASTING']] + hole_data['x'].tolist()
            y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
            z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
            
            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))

def update_figure_layout(fig):
    """Update the figure layout"""
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title="Elevation"
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0)
    )
def process_collar_data(collar_file):
    """Process collar file and return formatted DataFrame"""
    try:
        collar_df = read_mining_csv(collar_file)
        if collar_df is not None:
            st.write("Collar Data Preview:")
            st.write(collar_df.head())

            st.subheader("Select Collar Columns")
            hole_id_col = st.selectbox("Select HOLE_ID column", collar_df.columns)
            easting_col = st.selectbox("Select EASTING column", collar_df.columns)
            northing_col = st.selectbox("Select NORTHING column", collar_df.columns)
            elevation_col = st.selectbox("Select ELEVATION column", collar_df.columns)
            dip_col = st.selectbox("Select DIP column", collar_df.columns)
            azimuth_col = st.selectbox("Select AZIMUTH column", collar_df.columns)

            collar_df = collar_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                easting_col: 'EASTING',
                northing_col: 'NORTHING',
                elevation_col: 'ELEVATION',
                dip_col: 'DIP',
                azimuth_col: 'AZIMUTH'
            })
            
            # Convert numeric columns
            numeric_cols = ['EASTING', 'NORTHING', 'ELEVATION', 'DIP', 'AZIMUTH']
            for col in numeric_cols:
                collar_df[col] = pd.to_numeric(collar_df[col], errors='coerce')
                
            return collar_df
    except Exception as e:
        st.error(f"Error processing collar file: {str(e)}")
        return None

def process_assay_data(assay_file):
    """Process assay file and return formatted DataFrame"""
    try:
        assay_df = read_mining_csv(assay_file)
        if assay_df is not None:
            st.write("Assay Data Preview:")
            st.write(assay_df.head())

            st.subheader("Select Assay Columns")
            hole_id_col = st.selectbox("Select HOLE_ID column (Assay)", assay_df.columns)
            from_col = st.selectbox("Select FROM column", assay_df.columns)
            to_col = st.selectbox("Select TO column", assay_df.columns)
            
            element_cols = st.multiselect(
                "Select element columns",
                [col for col in assay_df.columns if col not in [hole_id_col, from_col, to_col]]
            )

            assay_df = assay_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                from_col: 'FROM',
                to_col: 'TO'
            })
            
            # Convert numeric columns
            numeric_cols = ['FROM', 'TO'] + element_cols
            for col in numeric_cols:
                assay_df[col] = pd.to_numeric(assay_df[col], errors='coerce')
            
            assay_df = assay_df[['HOLE_ID', 'FROM', 'TO'] + element_cols]
            return assay_df, element_cols
    except Exception as e:
        st.error(f"Error processing assay file: {str(e)}")
        return None, None

def process_litho_data(litho_file):
    """Process lithology file and return formatted DataFrame"""
    try:
        litho_df = read_mining_csv(litho_file)
        if litho_df is not None:
            st.write("Lithology Data Preview:")
            st.write(litho_df.head())

            st.subheader("Select Lithology Columns")
            hole_id_col = st.selectbox("Select HOLE_ID column (Lithology)", litho_df.columns)
            from_col = st.selectbox("Select FROM column (Lithology)", litho_df.columns)
            to_col = st.selectbox("Select TO column (Lithology)", litho_df.columns)
            litho_col = st.selectbox("Select LITHOLOGY column", litho_df.columns)

            litho_df = litho_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                from_col: 'FROM',
                to_col: 'TO',
                litho_col: 'LITHO'
            })
            
            # Convert numeric columns
            numeric_cols = ['FROM', 'TO']
            for col in numeric_cols:
                litho_df[col] = pd.to_numeric(litho_df[col], errors='coerce')
            
            litho_df = litho_df[['HOLE_ID', 'FROM', 'TO', 'LITHO']]
            return litho_df
    except Exception as e:
        st.error(f"Error processing lithology file: {str(e)}")
        return None
    

def show_statistical_analysis(merged_df, primary_element, use_log_scale):
    """Show detailed statistical analysis"""
    st.header("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Summary Statistics")
        stats_dict = {
            'Statistic': [
                'Count',
                'Mean',
                'Median',
                'Std Dev',
                'CV',
                'Min',
                'Q1',
                'Q3',
                'Max',
                'Skewness',
                'Kurtosis'
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
        stats_df['Value'] = stats_df['Value'].round(3)
        st.dataframe(stats_df.set_index('Statistic'))
        
    with col2:
        st.subheader("Distribution Plot")
        if use_log_scale and merged_df[primary_element].min() > 0:
            log_data = np.log10(merged_df[primary_element])
            fig = px.histogram(
                log_data,
                nbins=30,
                title=f"{primary_element} Distribution (log scale)"
            )
            # Calculate tick values for log scale
            min_log = np.floor(log_data.min())
            max_log = np.ceil(log_data.max())
            tick_vals = np.arange(min_log, max_log + 1)
            tick_text = [f'{10**val:.1g}' for val in tick_vals]
            
            fig.update_xaxes(
                title_text=primary_element,
                ticktext=tick_text,
                tickvals=tick_vals
            )
        else:
            fig = px.histogram(
                merged_df[primary_element],
                nbins=30,
                title=f"{primary_element} Distribution"
            )
            fig.update_xaxes(title_text=primary_element)
        
        st.plotly_chart(fig)

def calculate_significant_intervals(df, element, cutoff, min_length, max_internal_waste):
    """Calculate significant intervals based on parameters"""
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
            'last_significant_to': None  # Track the end of last significant sample
        }
        
        for idx, row in hole_data.iterrows():
            try:
                interval_length = float(row['TO']) - float(row['FROM'])
                current_grade = float(row[element])
                
                # Skip invalid intervals
                if interval_length <= 0 or pd.isna(current_grade):
                    continue
                
                if current_grade >= cutoff:
                    # If this is the first significant interval or if it continues from last significant
                    if (current_interval['start_depth'] is None or 
                        (current_interval['last_significant_to'] is not None and 
                         row['FROM'] - current_interval['last_significant_to'] <= max_internal_waste)):
                        
                        # Start new interval if none exists
                        if current_interval['start_depth'] is None:
                            current_interval['start_depth'] = row['FROM']
                        
                        # Add the waste interval if it exists
                        if (current_interval['last_significant_to'] is not None and 
                            row['FROM'] > current_interval['last_significant_to']):
                            waste_length = row['FROM'] - current_interval['last_significant_to']
                            current_interval['waste_lengths'].append(waste_length)
                        
                        # Add the significant interval
                        current_interval['grades'].append(current_grade)
                        current_interval['lengths'].append(interval_length)
                        if 'LITHO' in row:
                            current_interval['lithos'].append(row['LITHO'])
                        current_interval['end_depth'] = row['TO']
                        current_interval['last_significant_to'] = row['TO']
                    else:
                        # Start a new interval if previous one was closed
                        if current_interval['start_depth'] is not None:
                            # Close out previous interval
                            total_length = sum(current_interval['lengths'])
                            if total_length >= min_length:
                                weighted_grade = np.average(
                                    current_interval['grades'],
                                    weights=current_interval['lengths']
                                )
                                results.append({
                                    'HOLE_ID': hole_id,
                                    'FROM': current_interval['start_depth'],
                                    'TO': current_interval['end_depth'],
                                    'LENGTH': total_length,
                                    f'{element}_GRADE': weighted_grade,
                                    'INTERNAL_WASTE': sum(current_interval['waste_lengths']),
                                    'LITHOLOGY': ' / '.join(set(current_interval['lithos'])) if current_interval['lithos'] else None
                                })
                        
                        # Start new interval
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
                    # Check if this waste interval exceeds max_internal_waste
                    if row['FROM'] - current_interval['last_significant_to'] > max_internal_waste:
                        # Close out current interval
                        total_length = sum(current_interval['lengths'])
                        if total_length >= min_length:
                            weighted_grade = np.average(
                                current_interval['grades'],
                                weights=current_interval['lengths']
                            )
                            results.append({
                                'HOLE_ID': hole_id,
                                'FROM': current_interval['start_depth'],
                                'TO': current_interval['end_depth'],
                                'LENGTH': total_length,
                                f'{element}_GRADE': weighted_grade,
                                'INTERNAL_WASTE': sum(current_interval['waste_lengths']),
                                'LITHOLOGY': ' / '.join(set(current_interval['lithos'])) if current_interval['lithos'] else None
                            })
                        # Reset interval
                        current_interval = {
                            'start_depth': None,
                            'end_depth': None,
                            'grades': [],
                            'lengths': [],
                            'lithos': [],
                            'waste_lengths': [],
                            'last_significant_to': None
                        }
                
            except (ValueError, TypeError) as e:
                st.warning(f"Skipping invalid row in hole {hole_id}: {e}")
                continue
        
        # Close out final interval if exists
        if current_interval['start_depth'] is not None:
            total_length = sum(current_interval['lengths'])
            if total_length >= min_length:
                weighted_grade = np.average(
                    current_interval['grades'],
                    weights=current_interval['lengths']
                )
                results.append({
                    'HOLE_ID': hole_id,
                    'FROM': current_interval['start_depth'],
                    'TO': current_interval['end_depth'],
                    'LENGTH': total_length,
                    f'{element}_GRADE': weighted_grade,
                    'INTERNAL_WASTE': sum(current_interval['waste_lengths']),
                    'LITHOLOGY': ' / '.join(set(current_interval['lithos'])) if current_interval['lithos'] else None
                })
    
    return pd.DataFrame(results)
def show_download_options(merged_df, significant_intervals=None):
    """Show download options for processed data"""
    st.header("Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download processed data
        csv_processed = merged_df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data",
            data=csv_processed,
            file_name="processed_drillhole_data.csv",
            mime="text/csv"
        )
    
    with col2:
        if significant_intervals is not None:
            # Download significant intervals
            csv_intervals = significant_intervals.to_csv(index=False)
            st.download_button(
                label="Download Significant Intervals",
                data=csv_intervals,
                file_name="significant_intervals.csv",
                mime="text/csv"
            )    
# Main application title
st.title("3DA - Exploratory Data Analysis and Visualisation")

# File upload section
col1, col2, col3 = st.columns(3)

with col1:
    collar_file = st.file_uploader("Upload Collar File (CSV)", type=['csv'])

with col2:
    assay_file = st.file_uploader("Upload Assay File (CSV)", type=['csv'])

with col3:
    litho_file = st.file_uploader("Upload Lithology File (CSV)", type=['csv'])

# Validate data combinations
valid_data_combinations = False
if collar_file:
    if assay_file and not litho_file:
        valid_data_combinations = True
        analysis_mode = "collar_assay"
    elif litho_file and not assay_file:
        valid_data_combinations = True
        analysis_mode = "collar_litho"
    elif assay_file and litho_file:
        valid_data_combinations = True
        analysis_mode = "all"
else:
    st.warning("Collar file is required.")

if valid_data_combinations:
    # Process collar data
    collar_df = process_collar_data(collar_file)
    
    if collar_df is not None:
        # Initialise merged_df
        merged_df = None
        element_cols = []
        
        # Process based on analysis mode
        if analysis_mode in ["collar_assay", "all"]:
            assay_df, element_cols = process_assay_data(assay_file)
            if assay_df is not None:
                merged_df = pd.merge(collar_df, assay_df, on='HOLE_ID', how='inner')
        
        if analysis_mode in ["collar_litho", "all"]:
            litho_df = process_litho_data(litho_file)
            if litho_df is not None:
                if analysis_mode == "collar_litho":
                    merged_df = pd.merge(collar_df, litho_df, on='HOLE_ID', how='inner')
                elif merged_df is not None:  # all mode
                    merged_df = pd.merge_asof(
                        merged_df.sort_values('FROM'),
                        litho_df.sort_values('FROM'),
                        by='HOLE_ID',
                        on='FROM',
                        direction='nearest'
                    )
                    # Fix the TO column issue
                    merged_df = merged_df.rename(columns={'TO_x': 'TO'})
                    merged_df.drop('TO_y', axis=1, inplace=True)

        if merged_df is not None:
            # Add sidebar controls based on analysis mode
            st.sidebar.header("Analysis Controls")
            
            use_log_scale = st.sidebar.checkbox("Use log scale", value=True, key="main_log_scale")
            
            if analysis_mode in ["collar_assay", "all"]:
                primary_element = st.sidebar.selectbox("Select element for analysis:", element_cols)
                min_cutoff, max_cutoff = st.sidebar.slider(
                    f"{primary_element} cutoff range",
                    min_value=float(merged_df[primary_element].min()),
                    max_value=float(merged_df[primary_element].max()),
                    value=(float(merged_df[primary_element].min()), float(merged_df[primary_element].max())),
                    step=0.1
                )
                merged_df = merged_df[
                    (merged_df[primary_element] >= min_cutoff) & 
                    (merged_df[primary_element] <= max_cutoff)
                ]

            # Calculate 3D coordinates
            st.write("Calculating 3D coordinates...")
            
            # Calculate midpoints for assay/litho data
            if 'FROM' in merged_df.columns and 'TO' in merged_df.columns:
                merged_df['MIDPOINT'] = (merged_df['FROM'] + merged_df['TO']) / 2
            
            # Convert angles to radians
            merged_df['AZIMUTH_RAD'] = np.radians(90 - merged_df['AZIMUTH'])
            merged_df['DIP_RAD'] = np.radians(merged_df['DIP'])
            
            # Calculate direction components
            merged_df['dx'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.cos(merged_df['AZIMUTH_RAD'])
            merged_df['dy'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.sin(merged_df['AZIMUTH_RAD'])
            merged_df['dz'] = merged_df['MIDPOINT'] * np.sin(merged_df['DIP_RAD'])
            
            # Calculate final coordinates
            merged_df['x'] = merged_df['EASTING'] + merged_df['dx']
            merged_df['y'] = merged_df['NORTHING'] + merged_df['dy']
            merged_df['z'] = merged_df['ELEVATION'] + merged_df['dz']

            # Create 3D visualisation
            st.header("3D Drillhole visualisation")
            
            # Set visualisation mode based on analysis mode
            if analysis_mode == "collar_assay":
                viz_mode = "Grade"
            elif analysis_mode == "collar_litho":
                viz_mode = "Lithology"
            else:
                viz_mode = st.radio(
                    "visualisation Mode",
                    ["Grade", "Lithology", "Combined"],
                    horizontal=True
                )

            # Create figure
            fig = go.Figure()

            # Add visualisation based on mode
            if viz_mode in ["Grade", "Combined"] and analysis_mode in ["collar_assay", "all"]:
                add_grade_visualisation(fig, merged_df, primary_element, use_log_scale, viz_mode)

            if viz_mode in ["Lithology", "Combined"] and analysis_mode in ["collar_litho", "all"]:
                add_lithology_visualisation(fig, merged_df, viz_mode)

            # Add collar points
            add_collar_points(fig, collar_df)

            # Add drill traces
            add_drill_traces(fig, merged_df, collar_df, viz_mode)

            # Update layout
            update_figure_layout(fig)

            # Display the figure
            st.plotly_chart(fig)

            # Additional analysis sections
            if analysis_mode in ["collar_assay", "all"]:
                create_swath_plots(merged_df, primary_element, use_log_scale)
                show_statistical_analysis(merged_df, primary_element, use_log_scale)
                
                # Significant intervals calculation
                st.header("Significant Intervals")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_length = st.number_input(
                        "Minimum Interval Length (m)",
                        value=2.0,
                        min_value=0.1,
                        step=0.5
                    )
                
                with col2:
                    max_internal_waste = st.number_input(
                        "Maximum Internal Waste (m)",
                        value=2.0,
                        min_value=0.0,
                        step=0.5
                    )

                with col3:
                    interval_cutoff = st.number_input(
                        f"Minimum {primary_element} Grade",
                        value=float(merged_df[primary_element].median()),
                        min_value=float(merged_df[primary_element].min()),
                        max_value=float(merged_df[primary_element].max()),
                        step=0.1
                    )
                
                significant_intervals = None
                if st.button("Calculate Significant Intervals"):
                    significant_intervals = calculate_significant_intervals(
                        merged_df,
                        primary_element,
                        interval_cutoff,  # Use the new cutoff value
                        min_length,
                        max_internal_waste
                    )

                    if not significant_intervals.empty:
                        st.write(significant_intervals)
                    else:
                        st.warning("No significant intervals found with current parameters.")
            
            if analysis_mode in ["collar_litho", "all"]:
                create_lithology_analysis(merged_df, primary_element, use_log_scale)
            
            # Add download options section at the very end
            st.header("Download Options")
            
            # Create columns for download buttons
            download_cols = st.columns(4)
            
            # Processed Data
            with download_cols[0]:
                csv_processed = merged_df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv_processed,
                    file_name="processed_drillhole_data.csv",
                    mime="text/csv"
                )
            
            # Element Statistics (if available)
            if analysis_mode in ["collar_assay", "all"]:
                with download_cols[1]:
                    # Calculate element statistics
                    stats_dict = {
                        'Statistic': [
                            'Count',
                            'Mean',
                            'Median',
                            'Std Dev',
                            'CV',
                            'Min',
                            'Q1',
                            'Q3',
                            'Max',
                            'Skewness',
                            'Kurtosis'
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
                    stats_df['Value'] = stats_df['Value'].round(3)
                    
                    csv_stats = stats_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {primary_element} Statistics",
                        data=csv_stats,
                        file_name=f"{primary_element}_statistics.csv",
                        mime="text/csv"
                    )
                                # Significant Intervals (if calculated)
            if analysis_mode in ["collar_assay", "all"] and significant_intervals is not None and not significant_intervals.empty:
                with download_cols[2]:
                    csv_intervals = significant_intervals.to_csv(index=False)
                    st.download_button(
                        label="Download Significant Intervals",
                        data=csv_intervals,
                        file_name="significant_intervals.csv",
                        mime="text/csv"
                    )
            
            # Lithology Statistics (if available)
            if analysis_mode in ["collar_litho", "all"]:
                with download_cols[3]:
                    # Calculate lithology statistics
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
                            lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan  # CV
                        ]
                    })
                    
                    # Flatten column names and rename
                    litho_stats.columns = [
                        'Count', 'Mean', 'Median', 'Std Dev',
                        'Min', 'Q1', 'Q3', 'Max', 'CV'
                    ]
                    
                    # Reset index to make LITHO a column
                    litho_stats = litho_stats.reset_index()
                    
                    # Round statistics
                    numeric_columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Q1', 'Q3', 'Max', 'CV']
                    litho_stats[numeric_columns] = litho_stats[numeric_columns].round(3)
                    
                    csv_litho_stats = litho_stats.to_csv(index=False)
                    st.download_button(
                        label="Download Lithology Statistics",
                        data=csv_litho_stats,
                        file_name="lithology_statistics.csv",
                        mime="text/csv"
                    )