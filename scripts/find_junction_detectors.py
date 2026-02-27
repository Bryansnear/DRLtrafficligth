import sumolib
import xml.etree.ElementTree as ET

def find_junction_detectors(net_file, detector_file, junction_id):
    net = sumolib.net.readNet(net_file)
    junction = net.getNode(junction_id)
    incoming_lanes = []
    for edge in junction.getIncoming():
        for lane in edge.getLanes():
            incoming_lanes.append(lane.getID())
            
    print(f"Incoming lanes for junction {junction_id}: {incoming_lanes}")
    
    tree = ET.parse(detector_file)
    root = tree.getroot()
    
    relevant_detectors = []
    for detector in root.findall('laneAreaDetector'):
        lane_id = detector.get('lane')
        det_id = detector.get('id')
        if lane_id in incoming_lanes:
            relevant_detectors.append(det_id)
            
    print(f"Relevant detectors: {relevant_detectors}")

if __name__ == "__main__":
    net_file = "data/sumo/network/ibarra_mayorista.net.xml"
    detector_file = "data/sumo/additional/ibarra_mayorista.e2.xml"
    junction_id = "cluster_10018927538_10018927539_1236595970_7098850249_#1more"
    
    find_junction_detectors(net_file, detector_file, junction_id)
