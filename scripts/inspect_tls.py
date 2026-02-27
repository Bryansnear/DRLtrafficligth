import xml.etree.ElementTree as ET

NET_FILE = 'data/sumo/network/ibarra_mayorista_10cuadras.net.xml'
TLS_ID = '667932004'

tree = ET.parse(NET_FILE)
root = tree.getroot()

print(f"Inspecting TLS {TLS_ID}...")

# Inspect Logic/Phases
tls = root.find(f".//tlLogic[@id='{TLS_ID}']")
if tls:
    print(f"Type: {tls.get('type')}")
    print("Phases:")
    for i, p in enumerate(tls.findall('phase')):
        print(f"  Phase {i}: state='{p.get('state')}' duration={p.get('duration')}")
else:
    print("TLS Logic not found!")

# Inspect Connections to map Link IDs to Edges
print("\nConnections:")
connections = root.findall(f".//connection[@tl='{TLS_ID}']")
for c in connections:
    print(f"  Link {c.get('linkIndex')}: {c.get('from')} -> {c.get('to')} (dir {c.get('dir')})")
