import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# --- 1. UI Configuration ---
st.set_page_config(page_title="Interactive STING Clustering", layout="wide")
st.title("STING Algorithm: Interactive Grid Filtering")
st.markdown("Use the sidebar to change datasets and query the grid based on STING's pre-computed statistics.")

# --- 2. Robust Data Generation ---
@st.cache_data
def load_dataset(dataset_name):
    np.random.seed(42)
    if dataset_name == "Two Distinct Clusters":
        c1 = np.random.normal([1, 1], 0.3, (50, 2))
        c2 = np.random.normal([3, 3], 0.3, (50, 2))
        noise = np.random.uniform(0, 4, (15, 2))
        return np.clip(np.vstack((c1, c2, noise)), 0, 3.99)
        
    elif dataset_name == "Three Clusters + Heavy Noise":
        c1 = np.random.normal([0.5, 3.5], 0.2, (60, 2))
        c2 = np.random.normal([3.5, 0.5], 0.3, (50, 2))
        c3 = np.random.normal([2, 2], 0.15, (40, 2))
        noise = np.random.uniform(0, 4, (40, 2))
        return np.clip(np.vstack((c1, c2, c3, noise)), 0, 3.99)
        
    elif dataset_name == "Uniform Spread":
        return np.random.uniform(0, 4, (150, 2))

# --- 3. Pre-compute STING Statistics ---
@st.cache_data
def compute_grid_stats(data, grid_size=4):
    stats = []
    for x in range(grid_size):
        for y in range(grid_size):
            mask = (data[:, 0] >= x) & (data[:, 0] < x + 1) & (data[:, 1] >= y) & (data[:, 1] < y + 1)
            cell_data = data[mask]
            count = len(cell_data)
            
            if count > 0:
                mean_x, mean_y = np.mean(cell_data, axis=0)
                std_x, std_y = np.std(cell_data, axis=0)
            else:
                mean_x, mean_y, std_x, std_y = 0, 0, 0, 0
                
            stats.append({
                'x': x, 'y': y,
                'count': count,
                'mean_x': mean_x, 'mean_y': mean_y,
                'std_x': std_x, 'std_y': std_y,
                'std_total': np.sqrt(std_x**2 + std_y**2)
            })
    return pd.DataFrame(stats)

# --- 4. Sidebar Controls ---
st.sidebar.header("1. Choose Dataset")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset", 
    ["Two Distinct Clusters", "Three Clusters + Heavy Noise", "Uniform Spread"]
)

st.sidebar.header("2. STING Query Filter")
filter_type = st.sidebar.radio(
    "Filter Cells By:",
    ["Density (Count)", "Tightness (Std Dev)", "Location (Mean X)"]
)

data = load_dataset(dataset_choice)
grid_stats = compute_grid_stats(data)

if filter_type == "Density (Count)":
    threshold = st.sidebar.slider("Minimum Points per Cell", 1, 50, 5)
    valid_cells = grid_stats[grid_stats['count'] >= threshold]
    
elif filter_type == "Tightness (Std Dev)":
    threshold = st.sidebar.slider("Maximum Spread (Std Dev)", 0.0, 1.0, 0.3, 0.05)
    st.sidebar.caption("Lower = tighter clusters. Ignores empty cells.")
    valid_cells = grid_stats[(grid_stats['std_total'] <= threshold) & (grid_stats['count'] > 0)]
    
elif filter_type == "Location (Mean X)":
    min_x, max_x = st.sidebar.slider("Mean X Coordinate Range", 0.0, 4.0, (1.0, 3.0), 0.1)
    valid_cells = grid_stats[(grid_stats['mean_x'] >= min_x) & 
                             (grid_stats['mean_x'] <= max_x) & 
                             (grid_stats['count'] > 0)]

# --- 5. Clean 2D Plotly Visualization ---
fig_2d = go.Figure()

fig_2d.add_trace(go.Scatter(
    x=data[:, 0], y=data[:, 1],
    mode='markers',
    marker=dict(color='grey', size=6, opacity=0.6),
    hoverinfo='none'
))

shapes = []
for _, row in valid_cells.iterrows():
    x, y = row['x'], row['y']
    shapes.append(dict(
        type="rect",
        x0=x, y0=y, x1=x+1, y1=y+1,
        fillcolor="rgba(255, 0, 0, 0.2)",
        line=dict(color="red", width=2)
    ))
    
    hover_text = (f"<b>Cell [{x}, {y}]</b><br>Count: {row['count']}<br>"
                  f"Mean: ({row['mean_x']:.2f}, {row['mean_y']:.2f})<br>"
                  f"Std Dev: {row['std_total']:.2f}")
                  
    fig_2d.add_trace(go.Scatter(
        x=[x + 0.5], y=[y + 0.5],
        mode='text',
        text=[f"n={row['count']}"],
        textfont=dict(color='red', size=14),
        hovertext=hover_text,
        hoverinfo='text'
    ))

fig_2d.update_layout(
    shapes=shapes,
    xaxis=dict(range=[0, 4], dtick=1, showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(range=[0, 4], dtick=1, showgrid=True, gridcolor='lightgrey'),
    height=600, plot_bgcolor='white', showlegend=False,
    title="2D Grid-Based Statistical Query Results",
    margin=dict(l=40, r=40, t=60, b=40)
)

col1, col2 = st.columns([2, 1])
with col1:
    # Fix implemented: replaced use_container_width=True with width='stretch'
    st.plotly_chart(fig_2d, width='stretch')
with col2:
    st.write("### Filtered Cells Data")
    st.write(f"Found **{len(valid_cells)}** cells matching criteria.")
    st.dataframe(valid_cells[['x', 'y', 'count', 'mean_x', 'std_total']].round(2), hide_index=True)


# --- 6. The 3D Live Animation Engine ---
st.markdown("---")
st.write("### 3D Hierarchical Top-Down Simulation")
st.write("Watch how STING evaluates the grid from the top layer down, completely ignoring unpopulated quadrants.")

# Helper functions for Plotly 3D Drawing
def create_3d_grid(z_level, divisions, bounds, color):
    traces = []
    step = bounds / divisions
    for i in range(divisions + 1):
        loc = i * step
        traces.append(go.Scatter3d(x=[loc, loc], y=[0, bounds], z=[z_level, z_level], mode='lines', line=dict(color=color, width=4), hoverinfo='none'))
        traces.append(go.Scatter3d(x=[0, bounds], y=[loc, loc], z=[z_level, z_level], mode='lines', line=dict(color=color, width=4), hoverinfo='none'))
    return traces

def create_3d_cell(x, y, z, size, color):
    return go.Mesh3d(
        x=[x, x+size, x+size, x],
        y=[y, y, y+size, y+size],
        z=[z, z, z, z],
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=0.4, color=color, hoverinfo='none'
    )

if st.button("▶ Start 3D Simulation"):
    anim_placeholder = st.empty()
    
    # Pre-calculate Top-Down Logic based on Density threshold of 5
    THRESHOLD = 5
    valid_lvl2 = []
    valid_lvl3 = []
    
    for x2 in range(2):
        for y2 in range(2):
            pts = data[(data[:,0] >= x2*2) & (data[:,0] < x2*2+2) & (data[:,1] >= y2*2) & (data[:,1] < y2*2+2)]
            if len(pts) >= THRESHOLD: valid_lvl2.append((x2*2, y2*2))
                
    for (x_min, y_min) in valid_lvl2:
        for dx in range(2):
            for dy in range(2):
                pts = data[(data[:,0] >= x_min+dx) & (data[:,0] < x_min+dx+1) & (data[:,1] >= y_min+dy) & (data[:,1] < y_min+dy+1)]
                if len(pts) >= THRESHOLD: valid_lvl3.append((x_min+dx, y_min+dy))
    
    # Titles for each animation frame
    titles = [
        "Level 1 (Blue): Root Grid Evaluating Entire Map",
        "Level 2 (Green): Subdividing Map to 2x2",
        "Level 2 (Green): Isolating Valid Dense Quadrants",
        "Level 3 (Orange): Drilling ONLY into Valid Quadrants",
        "Level 3 (Red): Identifying Final Clusters",
        "Complete: Clusters Projected down to Raw Data"
    ]
    
    # Animation Loop
    for frame in range(6):
        fig_3d = go.Figure()
        
        # Base Data Points (Always visible)
        pt_colors = 'grey'
        if frame == 5:
            # Color points in final frame
            pt_colors = ['red' if any(x <= p[0] < x+1 and y <= p[1] < y+1 for x,y in valid_lvl3) else 'grey' for p in data]
            
        fig_3d.add_trace(go.Scatter3d(x=data[:,0], y=data[:,1], z=np.zeros(len(data)), mode='markers', marker=dict(size=4, color=pt_colors), hoverinfo='none'))
        
        # Build Layers incrementally based on current frame
        if frame >= 0: fig_3d.add_traces(create_3d_grid(3, 1, 4, 'blue'))
        if frame >= 1: fig_3d.add_traces(create_3d_grid(2, 2, 4, 'green'))
        if frame >= 2: 
            for (x,y) in valid_lvl2: fig_3d.add_trace(create_3d_cell(x, y, 2, 2, 'green'))
        if frame >= 3:
            for (x,y) in valid_lvl2:
                fig_3d.add_traces(create_3d_grid(1, 4, 4, 'orange')) # Simplified grid for visual
        if frame >= 4:
            for (x,y) in valid_lvl3: fig_3d.add_trace(create_3d_cell(x, y, 1, 1, 'red'))
            
        fig_3d.update_layout(
            title=dict(text=f"Frame {frame+1}/6: {titles[frame]}", font=dict(size=18)),
            scene=dict(
                xaxis=dict(range=[0, 4]), yaxis=dict(range=[0, 4]), zaxis=dict(range=[0, 3.5]),
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)) # Isometric viewing angle
            ),
            height=700, showlegend=False, margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Update placeholder in real-time
        anim_placeholder.plotly_chart(fig_3d, width='stretch')
        time.sleep(1.2) # Wait 1.2 seconds before showing next frame