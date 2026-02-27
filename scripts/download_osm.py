import requests
import os

def download_osm(south, west, north, east, output_file):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:xml];
    (
      node({south},{west},{north},{east});
      way({south},{west},{north},{east});
      relation({south},{west},{north},{east});
    );
    out meta;
    >;
    out meta qt;
    """
    
    print(f"Downloading OSM data for bbox: {south}, {west}, {north}, {east}")
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Successfully saved to {output_file}")
    else:
        print(f"Error downloading data: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Coordenadas EXACTAS del usuario desde OpenStreetMap
    # Centro: 0.361102, -78.122752 (zoom 18)
    # Mercado Mayorista Ibarra - grilla urbana
    
    lat = 0.361102
    lon = -78.122752
    delta = 0.004  # ~450m en cada dirección = ~900m total (~10 cuadras)
    
    south = lat - delta
    north = lat + delta
    west = lon - delta
    east = lon + delta
    
    output_dir = r"c:\Users\bryan\Maestria\Tesis\Desarollo\Cursor gpt5\data\sumo\network"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ibarra_mayorista_10cuadras.osm")
    
    download_osm(south, west, north, east, output_file)
