"""
Script para generar detectores E2 SOLO en la calle 13 de Abril.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import sumolib

OSM_FILE = Path("data/sumo/network/ibarra_mayorista_con_tls.osm")
NET_FILE = Path("data/sumo/network/ibarra_mayorista_10cuadras.net.xml")
OUTPUT_DIR = Path("data/sumo")

def find_13_abril_way_ids(osm_file):
    """Encuentra los IDs de los ways que son la calle 13 de Abril."""
    tree = ET.parse(osm_file)
    root = tree.getroot()
    
    way_ids = []
    for way in root.findall('way'):
        name_tag = way.find("tag[@k='name']")
        if name_tag is not None:
            name = name_tag.get('v', '').lower()
            if '13 de abril' in name:
                way_ids.append(way.get('id'))
                print(f"Encontrado way: {way.get('id')} - {name_tag.get('v')}")
    
    return way_ids

def find_edges_from_ways(net_file, way_ids):
    """Encuentra los edges de SUMO que corresponden a los ways de OSM."""
    net = sumolib.net.readNet(str(net_file))
    
    matching_edges = []
    for edge in net.getEdges():
        edge_id = edge.getID()
        # Los IDs de edges en SUMO suelen contener el way ID de OSM
        for way_id in way_ids:
            if way_id in edge_id or edge_id.startswith(way_id) or edge_id.endswith(way_id):
                matching_edges.append(edge)
                break
        # También buscar por nombre si está disponible
        if edge.getName():
            if '13 de abril' in edge.getName().lower():
                if edge not in matching_edges:
                    matching_edges.append(edge)
    
    return matching_edges

def generate_e2_detectors_for_edges(edges, output_file):
    """Genera detectores E2 para los edges especificados."""
    root = ET.Element("additional")
    
    detector_count = 0
    e2_ids = []
    
    for edge in edges:
        edge_id = edge.getID()
        length = min(edge.getLength() - 5, 100)
        
        if length < 10:
            continue
        
        for lane_idx in range(edge.getLaneNumber()):
            lane_id = f"{edge_id}_{lane_idx}"
            det_id = f"e2_13abril_{edge_id}_{lane_idx}"
            
            e2 = ET.SubElement(root, "laneAreaDetector")
            e2.set("id", det_id)
            e2.set("lane", lane_id)
            e2.set("pos", "0")
            e2.set("endPos", str(length))
            e2.set("freq", "5")
            e2.set("file", "e2_output.xml")
            e2.set("friendlyPos", "true")
            
            e2_ids.append(det_id)
            detector_count += 1
            print(f"  Detector: {det_id}")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"\nGenerados {detector_count} detectores E2 en {output_file}")
    return e2_ids

def main():
    print("=" * 60)
    print("GENERANDO DETECTORES E2 SOLO EN CALLE 13 DE ABRIL")
    print("=" * 60)
    
    # Buscar ways de 13 de Abril en OSM
    print("\n[1] Buscando calle 13 de Abril en OSM...")
    way_ids = find_13_abril_way_ids(OSM_FILE)
    print(f"Ways encontrados: {len(way_ids)}")
    
    # Buscar edges correspondientes en la red SUMO
    print("\n[2] Buscando edges en la red SUMO...")
    net = sumolib.net.readNet(str(NET_FILE))
    
    # Buscar todos los edges y filtrar por nombre o por la orientación
    # La calle 13 de Abril parece ser vertical en el mapa (norte-sur)
    edges_13_abril = []
    
    for edge in net.getEdges():
        edge_id = edge.getID()
        # Buscar edges que contengan alguno de los way IDs
        for way_id in way_ids:
            if way_id in edge_id:
                edges_13_abril.append(edge)
                print(f"  Edge encontrado: {edge_id}")
                break
    
    if not edges_13_abril:
        print("\nNo se encontraron edges directamente. Buscando por posición...")
        # Si no encontramos por ID, buscar edges en la zona central (donde está 13 de Abril)
        # Basado en las coordenadas, la 13 de Abril está aproximadamente en x=500-600
        for edge in net.getEdges():
            if not edge.allows("passenger"):
                continue
            # Obtener posición del edge
            shape = edge.getShape()
            if shape:
                x_coords = [p[0] for p in shape]
                avg_x = sum(x_coords) / len(x_coords)
                # La calle 13 de Abril está aproximadamente en x=480-550
                if 480 <= avg_x <= 550:
                    # Verificar que es vertical (diferencia en Y mayor que en X)
                    y_coords = [p[1] for p in shape]
                    dx = max(x_coords) - min(x_coords)
                    dy = max(y_coords) - min(y_coords)
                    if dy > dx * 2:  # Es más vertical que horizontal
                        edges_13_abril.append(edge)
                        print(f"  Edge vertical en 13 de Abril: {edge.getID()}")
    
    print(f"\nTotal edges en 13 de Abril: {len(edges_13_abril)}")
    
    # Generar detectores
    print("\n[3] Generando detectores E2...")
    output_file = OUTPUT_DIR / "additional" / "ibarra_13abril_e2.add.xml"
    e2_ids = generate_e2_detectors_for_edges(edges_13_abril, output_file)
    
    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()
