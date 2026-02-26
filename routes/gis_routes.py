from flask import Blueprint, render_template
import pandas as pd
import folium

gis_bp = Blueprint('gis', __name__)

@gis_bp.route('/map')
def map_view():
    # Load data
    df = pd.read_csv("data/area_waste_locations.csv")

    # Create map centered around average coordinates
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

    # Add markers
    for _, row in df.iterrows():
        # Color code based on waste quantity
        if row['waste_kg'] < 300:
            color = 'green'
        elif row['waste_kg'] < 500:
            color = 'orange'
        else:
            color = 'red'

        popup_text = f"<b>{row['area_name']}</b><br>Waste: {row['waste_kg']} kg"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    # Save the map
    m.save("templates/map.html")
    return render_template("map.html")
