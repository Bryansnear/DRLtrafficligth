"""
Configuración específica para la intersección 667932004.
Crea detectores E2 para las 4 vías entrantes, similar al escenario de entrenamiento.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import sumolib

NET_FILE = Path("data/sumo/network/ibarra_mayorista_10cuadras.net.xml")
OUTPUT_DIR = Path("data/sumo")

# Intersección objetivo
TARGET_NODE = "667932004"

def setup_intersection_detectors():
    """Configura detectores E2 para la intersección 667932004."""
    
    print("=" * 60)
    print(f"CONFIGURANDO INTERSECCIÓN {TARGET_NODE}")
    print("=" * 60)
    
    net = sumolib.net.readNet(str(NET_FILE))
    node = net.getNode(TARGET_NODE)
    
    print(f"\nNodo: {node.getID()}")
    print(f"Tipo: {node.getType()}")
    print(f"Coordenadas: {node.getCoord()}")
    
    # Obtener edges entrantes
    incoming = list(node.getIncoming())
    print(f"\nEdges entrantes ({len(incoming)}):")
    
    # Crear archivo de detectores
    root = ET.Element("additional")
    
    e2_ids = []
    for i, edge in enumerate(incoming):
        edge_id = edge.getID()
        length = edge.getLength()
        lanes = edge.getLaneNumber()
        
        print(f"  [{i}] {edge_id}: {length:.1f}m, {lanes} carril(es)")
        
        # Determinar dirección basado en el nombre del edge
        if "1228099685" in edge_id:
            direction = "N"  # Norte
        elif "756028261" in edge_id:
            direction = "S"  # Sur
        elif "1228099684" in edge_id:
            direction = "E"  # Este
        elif "51962697" in edge_id:
            direction = "W"  # Oeste
        else:
            direction = f"D{i}"
        
        # Crear detector para cada carril
        for lane_idx in range(lanes):
            lane_id = f"{edge_id}_{lane_idx}"
            det_id = f"{direction}_in_{lane_idx}"
            
            # Longitud del detector (máximo 50m o longitud del edge - 5m)
            det_length = min(length - 5, 50)
            if det_length < 10:
                det_length = length - 2  # Usar casi todo el edge si es muy corto
            
            e2 = ET.SubElement(root, "laneAreaDetector")
            e2.set("id", det_id)
            e2.set("lane", lane_id)
            e2.set("pos", str(max(0, length - det_length - 2)))  # Empezar antes del final
            e2.set("endPos", str(length - 2))  # Terminar 2m antes del semáforo
            e2.set("freq", "5")
            e2.set("file", "e2_output.xml")
            e2.set("friendlyPos", "true")
            
            e2_ids.append(det_id)
            print(f"      Detector: {det_id} en {lane_id}")
    
    # Guardar archivo de detectores
    output_file = OUTPUT_DIR / "additional" / "intersection_667932004_e2.add.xml"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    print(f"\nDetectores guardados en: {output_file}")
    print(f"IDs de detectores: {e2_ids}")
    
    # Crear archivo de configuración SUMO específico
    create_sumo_config(e2_ids)
    
    return e2_ids

def create_sumo_config(e2_ids):
    """Crea archivo de configuración SUMO para la intersección."""
    
    root = ET.Element("configuration")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    
    # Input
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file").set("value", "../network/ibarra_mayorista_10cuadras.net.xml")
    ET.SubElement(input_elem, "route-files").set("value", "../routes/ibarra_mayorista_routes.rou.xml")
    ET.SubElement(input_elem, "additional-files").set("value", "../additional/intersection_667932004_e2.add.xml")
    
    # Time
    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin").set("value", "0")
    ET.SubElement(time_elem, "end").set("value", "7200")
    ET.SubElement(time_elem, "step-length").set("value", "1")
    
    # Processing
    proc_elem = ET.SubElement(root, "processing")
    ET.SubElement(proc_elem, "ignore-route-errors").set("value", "true")
    
    # Report
    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "verbose").set("value", "false")
    ET.SubElement(report_elem, "no-step-log").set("value", "true")
    
    # GUI
    gui_elem = ET.SubElement(root, "gui_only")
    ET.SubElement(gui_elem, "start").set("value", "false")
    ET.SubElement(gui_elem, "quit-on-end").set("value", "false")
    
    output_file = OUTPUT_DIR / "cfg" / "intersection_667932004.sumocfg"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    print(f"Config SUMO guardado en: {output_file}")

if __name__ == "__main__":
    e2_ids = setup_intersection_detectors()
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN COMPLETADA")
    print("=" * 60)
    print(f"\nPara usar en evaluación, los detectores son:")
    print(f"  e2_ids = {e2_ids}")
