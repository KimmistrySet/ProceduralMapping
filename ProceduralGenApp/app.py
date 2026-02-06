import streamlit as st
import numpy as np
import noise
from PIL import Image
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="MindFull Terrain Gen", layout="wide")

st.title("MindFull Studios: Procedural Environment Generator")
st.markdown("### The Voice-Ready Terrain Engine")

# --- SIDEBAR CONTROLS (Replacing your HTML Controls) ---
with st.sidebar:
    st.header("Generation Parameters")
    seed = st.number_input("Seed", value=42, step=1)
    scale = st.slider("Scale", 10.0, 300.0, 100.0, step=5.0)
    octaves = st.slider("Octaves", 1, 10, 6)
    persistence = st.slider("Persistence", 0.1, 1.0, 0.5)
    lacunarity = st.slider("Lacunarity", 1.0, 4.0, 2.0)
    
    st.markdown("---")
    st.info("Voice Input Module: [Offline]")

# --- YOUR ORIGINAL LOGIC (Refactored to remove Supabase) ---
def generate_maps(seed, scale, octaves, persistence, lacunarity, map_size=512):
    # 1. Generate Multi-Octave Perlin Noise
    world = np.zeros((map_size, map_size))
    for i in range(map_size):
        for j in range(map_size):
            x = (i / map_size) * scale
            y = (j / map_size) * scale
            world[i][j] = noise.pnoise2(x, y, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=scale, 
                                        repeaty=scale, 
                                        base=seed)
    
    # 2. Normalize
    min_val = np.min(world)
    max_val = np.max(world)
    normalized_height = (world - min_val) / (max_val - min_val)
    height_map_array = (normalized_height * 255).astype(np.uint8)
    
    # 3. Color Banding (Your exact colors)
    color_bands = [
        (0.00, (10, 10, 80)), (0.05, (60, 120, 180)), (0.10, (210, 210, 120)),
        (0.45, (80, 150, 40)), (0.70, (150, 150, 150)), (1.00, (240, 240, 240))
    ]
    color_map_array = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    
    for i in range(len(color_bands)):
        threshold, color = color_bands[i]
        lower_threshold = color_bands[i-1][0] if i > 0 else 0.0
        mask = (normalized_height >= lower_threshold) & (normalized_height < threshold)
        color_map_array[mask] = color
        
    return height_map_array, color_map_array

# --- EXECUTION ---
if st.button("Generate Terrain", type="primary"):
    with st.spinner("Calculating Perlin Noise..."):
        # Run the logic
        h_map, c_map = generate_maps(seed, scale, octaves, persistence, lacunarity)
        
        # Convert to Images
        height_img = Image.fromarray(h_map, mode='L')
        color_img = Image.fromarray(c_map, mode='RGB')
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Color Map")
            st.image(color_img, use_container_width=True)
        with col2:
            st.subheader("Height Map")
            st.image(height_img, use_container_width=True)

        # Download Logic
        buf = io.BytesIO()
        color_img.save(buf, format="PNG")
        st.download_button("Download Color Map", data=buf.getvalue(), file_name=f"terrain_{seed}.png", mime="image/png")