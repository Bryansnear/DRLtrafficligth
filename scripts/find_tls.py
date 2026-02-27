import sumolib

def find_tls_junctions(net_file):
    net = sumolib.net.readNet(net_file)
    tls_junctions = []
    for node in net.getNodes():
        if node.getType() == "traffic_light":
            tls_junctions.append({
                "id": node.getID(),
                "coord": node.getCoord(),
                "incoming": [e.getID() for e in node.getIncoming()],
                "outgoing": [e.getID() for e in node.getOutgoing()]
            })
    
    print(f"Found {len(tls_junctions)} traffic light junctions:")
    for tls in tls_junctions:
        print(f"ID: {tls['id']}, Coord: {tls['coord']}, Incoming: {len(tls['incoming'])}, Outgoing: {len(tls['outgoing'])}")

if __name__ == "__main__":
    net_file = "data/sumo/network/ibarra_mayorista_10cuadras.net.xml"
    find_tls_junctions(net_file)
