import os
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def generate_dynamic_route_file(output_path: str, total_time: int = 3600):
    """
    Generates a SUMO route file with dynamic, PROBABILISTIC traffic patterns.
    Uses FLOWS instead of fixed vehicles for true randomness across episodes.
    
    Phases:
    1. 0-15 min: Medium Balanced (Warmup)
    2. 15-30 min: South Peak (High S, Low E)
    3. 30-40 min: Medium Balanced (Transition)
    4. 40-55 min: East Peak (High E, Low S)
    5. 55-60 min: Alternating Bursts
    """
    
    with open(output_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15.0" guiShape="passenger"/>\n\n')
        
        # Define routes
        f.write('    <!-- Routes based on network topology (E_to_C, S_to_C) -->\n')
        f.write('    <route id="r_S_N" edges="S_to_C C_to_N"/>\n')
        f.write('    <route id="r_S_W" edges="S_to_C C_to_W"/>\n')
        f.write('    <route id="r_E_W" edges="E_to_C C_to_W"/>\n')
        f.write('    <route id="r_E_N" edges="E_to_C C_to_N"/>\n\n')
        
        # Phase 1: Warmup (0-900s) - Medium Balanced
        f.write('    <!-- Phase 1: Warmup (0-15 min) - Medium Balanced -->\n')
        f.write('    <flow id="S_N_warmup" type="car" route="r_S_N" begin="0" end="900" probability="0.13" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="S_W_warmup" type="car" route="r_S_W" begin="0" end="900" probability="0.03" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_W_warmup" type="car" route="r_E_W" begin="0" end="900" probability="0.13" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_N_warmup" type="car" route="r_E_N" begin="0" end="900" probability="0.03" departLane="best" departSpeed="max"/>\n\n')
        
        # Phase 2: South Peak (900-1800s)
        f.write('    <!-- Phase 2: South Peak (15-30 min) - High S, Low E -->\n')
        f.write('    <flow id="S_N_peak" type="car" route="r_S_N" begin="900" end="1800" probability="0.40" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="S_W_peak" type="car" route="r_S_W" begin="900" end="1800" probability="0.10" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_W_low" type="car" route="r_E_W" begin="900" end="1800" probability="0.06" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_N_low" type="car" route="r_E_N" begin="900" end="1800" probability="0.02" departLane="best" departSpeed="max"/>\n\n')
        
        # Phase 3: Transition (1800-2400s)
        f.write('    <!-- Phase 3: Transition (30-40 min) - Medium Balanced -->\n')
        f.write('    <flow id="S_N_trans" type="car" route="r_S_N" begin="1800" end="2400" probability="0.13" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="S_W_trans" type="car" route="r_S_W" begin="1800" end="2400" probability="0.03" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_W_trans" type="car" route="r_E_W" begin="1800" end="2400" probability="0.13" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_N_trans" type="car" route="r_E_N" begin="1800" end="2400" probability="0.03" departLane="best" departSpeed="max"/>\n\n')
        
        # Phase 4: East Peak (2400-3300s)
        f.write('    <!-- Phase 4: East Peak (40-55 min) - High E, Low S -->\n')
        f.write('    <flow id="S_N_low2" type="car" route="r_S_N" begin="2400" end="3300" probability="0.06" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="S_W_low2" type="car" route="r_S_W" begin="2400" end="3300" probability="0.02" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_W_peak2" type="car" route="r_E_W" begin="2400" end="3300" probability="0.40" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_N_peak2" type="car" route="r_E_N" begin="2400" end="3300" probability="0.10" departLane="best" departSpeed="max"/>\n\n')
        
        # Phase 5: Bursts (3300-3600s) - Alternating
        f.write('    <!-- Phase 5: Bursts (55-60 min) - High variability -->\n')
        f.write('    <flow id="S_burst" type="car" route="r_S_N" begin="3300" end="3600" probability="0.25" departLane="best" departSpeed="max"/>\n')
        f.write('    <flow id="E_burst" type="car" route="r_E_W" begin="3300" end="3600" probability="0.25" departLane="best" departSpeed="max"/>\n\n')
        
        f.write('</routes>\n')
    
    print(f"Generated probabilistic dynamic route file at: {output_path}")
    print(f"Using FLOWS for true randomness across episodes")

if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "data" / "sumo" / "routes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "dynamic_wave.rou.xml"
    generate_dynamic_route_file(str(output_file))
