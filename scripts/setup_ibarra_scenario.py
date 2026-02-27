"""
Script para generar la configuración completa del escenario Ibarra Mayorista:
1. Rutas de vehículos para todo el mapa
2. Detectores E2 en la calle 13 de Abril
3. Archivos de configuración SUMO
"""
import sumolib
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Rutas
NET_FILE = Path("data/sumo/network/ibarra_mayorista_10cuadras.net.xml")
OUTPUT_DIR = Path("data/sumo")

def generate_routes(net_file, output_file, duration=3600, veh_per_hour=500):
    """Genera rutas aleatorias para vehículos en toda la red."""
    net = sumolib.net.readNet(str(net_file))
    edges = net.getEdges()
    
    # Filtrar edges válidos (con carriles para vehículos)
    valid_edges = [e for e in edges if e.allows("passenger") and e.getLength() > 10]
    
    print(f"Edges válidos para rutas: {len(valid_edges)}")
    
    # Generar rutas XML
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Tipo de vehículo
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6")
    vtype.set("decel", "4.5")
    vtype.set("sigma", "0.5")
    vtype.set("length", "4.5")
    vtype.set("maxSpeed", "15")
    vtype.set("color", "0.3,0.3,1.0")
    
    # Calcular intervalo entre vehículos
    interval = 3600 / veh_per_hour
    
    veh_id = 0
    for t in range(0, duration, int(interval)):
        # Seleccionar origen y destino aleatorios
        from_edge = random.choice(valid_edges)
        to_edge = random.choice(valid_edges)
        
        # Evitar mismo origen y destino
        attempts = 0
        while from_edge == to_edge and attempts < 10:
            to_edge = random.choice(valid_edges)
            attempts += 1
        
        if from_edge != to_edge:
            trip = ET.SubElement(root, "trip")
            trip.set("id", f"veh_{veh_id}")
            trip.set("type", "car")
            trip.set("depart", str(t + random.uniform(0, interval)))
            trip.set("from", from_edge.getID())
            trip.set("to", to_edge.getID())
            veh_id += 1
    
    # Guardar archivo
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generadas {veh_id} rutas en {output_file}")
    return veh_id

def generate_e2_detectors(net_file, output_file, target_tls_ids=None):
    """
    Genera detectores E2 para los semáforos especificados.
    Si target_tls_ids es None, genera para todos los semáforos.
    """
    net = sumolib.net.readNet(str(net_file))
    
    root = ET.Element("additional")
    
    # Obtener todos los nodos tipo semáforo
    tls_nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    
    if target_tls_ids:
        tls_nodes = [n for n in tls_nodes if n.getID() in target_tls_ids]
    
    print(f"Generando detectores E2 para {len(tls_nodes)} semáforos")
    
    detector_count = 0
    e2_ids = []
    
    for node in tls_nodes:
        # Para cada edge entrante al semáforo
        for edge in node.getIncoming():
            if not edge.allows("passenger"):
                continue
                
            edge_id = edge.getID()
            length = min(edge.getLength() - 5, 100)  # Max 100m, dejar 5m antes del semáforo
            
            if length < 10:  # Mínimo 10m
                continue
            
            for lane_idx in range(edge.getLaneNumber()):
                lane_id = f"{edge_id}_{lane_idx}"
                det_id = f"e2_{edge_id}_{lane_idx}"
                
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
    
    # Guardar archivo
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generados {detector_count} detectores E2 en {output_file}")
    return e2_ids

def generate_sumo_config(net_file, routes_file, additional_file, output_file, duration=3600):
    """Genera archivo de configuración SUMO."""
    root = ET.Element("configuration")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    
    # Input
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file").set("value", f"../network/{Path(net_file).name}")
    ET.SubElement(input_elem, "route-files").set("value", f"../routes/{Path(routes_file).name}")
    ET.SubElement(input_elem, "additional-files").set("value", f"../additional/{Path(additional_file).name}")
    
    # Time
    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin").set("value", "0")
    ET.SubElement(time_elem, "end").set("value", str(duration))
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
    ET.SubElement(gui_elem, "start").set("value", "true")
    ET.SubElement(gui_elem, "quit-on-end").set("value", "false")
    
    # Guardar
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Configuración SUMO guardada en {output_file}")

def main():
    print("=" * 60)
    print("CONFIGURANDO ESCENARIO IBARRA MAYORISTA")
    print("=" * 60)
    
    # Crear directorios si no existen
    (OUTPUT_DIR / "routes").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "additional").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cfg").mkdir(parents=True, exist_ok=True)
    
    # 1. Generar rutas
    print("\n[1/3] Generando rutas de vehículos...")
    routes_file = OUTPUT_DIR / "routes" / "ibarra_mayorista_routes.rou.xml"
    generate_routes(NET_FILE, routes_file, duration=3600, veh_per_hour=1500)
    
    # 2. Generar detectores E2
    # Semáforos principales en la zona de la calle 13 de Abril (basado en coordenadas)
    print("\n[2/3] Generando detectores E2...")
    
    # Estos son los semáforos que parecen estar en la calle 13 de Abril 
    # basado en las coordenadas (columna central del mapa)
    target_tls = [
        "667931952",  # (445.94, 302.35)
        "667931948",  # (498.28, 273.61)
        "667931949",  # (546.98, 239.36)
        "1234094610", # (332.65, 354.38)
        "667931953",  # (336.47, 362.15)
        "1237551450", # (385.75, 325.72)
        "662697202",  # (390.73, 332.27)
        "1237551454", # (440.04, 296.42)
        "1234094612", # (493.94, 266.97)
        "1234094618", # (550.92, 246.3)
    ]
    
    additional_file = OUTPUT_DIR / "additional" / "ibarra_mayorista_e2.add.xml"
    e2_ids = generate_e2_detectors(NET_FILE, additional_file, target_tls_ids=target_tls)
    
    # 3. Generar configuración SUMO
    print("\n[3/3] Generando configuración SUMO...")
    config_file = OUTPUT_DIR / "cfg" / "ibarra_mayorista_baseline.sumocfg"
    generate_sumo_config(NET_FILE, routes_file, additional_file, config_file)
    
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN COMPLETADA")
    print("=" * 60)
    print(f"\nArchivos generados:")
    print(f"  - Rutas: {routes_file}")
    print(f"  - Detectores E2: {additional_file}")
    print(f"  - Config SUMO: {config_file}")
    print(f"\nDetectores E2 generados: {len(e2_ids)}")
    print(f"\nPara ejecutar con GUI:")
    print(f"  sumo-gui -c {config_file}")
    
    # Guardar lista de detectores E2 para el config YAML
    print(f"\nIDs de detectores E2 para YAML config:")
    print(f"  e2_ids: {e2_ids[:10]}...")  # Primeros 10

if __name__ == "__main__":
    random.seed(42)
    main()
