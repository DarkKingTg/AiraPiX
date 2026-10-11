import xml.etree.ElementTree as ET
from pathlib import Path

css_animations = """
    @keyframes flowAnim {
      from { stroke-dashoffset: 20; }
      to { stroke-dashoffset: 0; }
    }
    @keyframes cyanGlow {
      0%, 100% { filter: drop-shadow(0 0 2px rgba(0, 242, 254, 0.3)); }
      50% { filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.75)); }
    }
    @keyframes greenGlow {
      0%, 100% { filter: drop-shadow(0 0 2px rgba(16, 185, 129, 0.3)); }
      50% { filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.75)); }
    }
    @keyframes purpleGlow {
      0%, 100% { filter: drop-shadow(0 0 2px rgba(196, 113, 237, 0.3)); }
      50% { filter: drop-shadow(0 0 10px rgba(196, 113, 237, 0.75)); }
    }
    @keyframes amberGlow {
      0%, 100% { filter: drop-shadow(0 0 2px rgba(245, 158, 11, 0.3)); }
      50% { filter: drop-shadow(0 0 10px rgba(245, 158, 11, 0.75)); }
    }
    @keyframes corePulse {
      0%, 100% { filter: drop-shadow(0 0 3px rgba(99, 102, 241, 0.4)); }
      50% { filter: drop-shadow(0 0 14px rgba(99, 102, 241, 0.9)); }
    }
    .flow-cyan { stroke-dasharray: 6, 4; animation: flowAnim 1.2s linear infinite; }
    .flow-emerald { stroke-dasharray: 6, 4; animation: flowAnim 1.2s linear infinite; }
    .flow-purple { stroke-dasharray: 6, 4; animation: flowAnim 1.2s linear infinite; }
    .flow-amber { stroke-dasharray: 6, 4; animation: flowAnim 1.2s linear infinite; }
    .glow-cyan { animation: cyanGlow 4s ease-in-out infinite; }
    .glow-emerald { animation: greenGlow 4s ease-in-out infinite; }
    .glow-purple { animation: purpleGlow 4s ease-in-out infinite; }
    .glow-amber { animation: amberGlow 4s ease-in-out infinite; }
    .glow-core { animation: corePulse 3.5s ease-in-out infinite; }
"""

for file_path in [Path("docs/airapix_architecture.svg"), Path("Report/airapix_architecture.svg")]:
    txt = file_path.read_text(encoding="utf-8")
    if "flowAnim" not in txt:
        txt = txt.replace("  <style>", "  <style>" + css_animations)

    # Add glow classes to stage rects
    txt = txt.replace('stroke="#00f2fe" stroke-width="1.5"/>', 'stroke="#00f2fe" stroke-width="1.5" class="glow-cyan"/>', 1)
    txt = txt.replace('stroke="#10b981" stroke-width="1.5"/>', 'stroke="#10b981" stroke-width="1.5" class="glow-emerald"/>', 1)
    txt = txt.replace('stroke="#c471ed" stroke-width="1.5"/>', 'stroke="#c471ed" stroke-width="1.5" class="glow-purple"/>', 1)
    txt = txt.replace('stroke="#38bdf8" stroke-width="1.5"/>', 'stroke="#38bdf8" stroke-width="1.5" class="glow-cyan"/>', 1)
    txt = txt.replace('stroke="#6366f1" stroke-width="2"/>', 'stroke="#6366f1" stroke-width="2" class="glow-core"/>', 1)
    txt = txt.replace('stroke="#f59e0b" stroke-width="1.5"/>', 'stroke="#f59e0b" stroke-width="1.5" class="glow-amber"/>', 1)
    txt = txt.replace('stroke="#a855f7" stroke-width="1.5"/>', 'stroke="#a855f7" stroke-width="1.5" class="glow-purple"/>', 1)

    # Animate connecting flow arrows
    txt = txt.replace('d="M 315 275 L 315 305" stroke="#00f2fe" stroke-width="2" stroke-dasharray="4,4"', 'd="M 315 275 L 315 305" stroke="#00f2fe" stroke-width="2" class="flow-cyan"')
    txt = txt.replace('d="M 885 275 L 885 305" stroke="#10b981" stroke-width="2" stroke-dasharray="4,4"', 'd="M 885 275 L 885 305" stroke="#10b981" stroke-width="2" class="flow-emerald"')
    txt = txt.replace('d="M 315 480 L 315 508" stroke="#c471ed" stroke-width="2"', 'd="M 315 480 L 315 508" stroke="#c471ed" stroke-width="2" class="flow-purple"')
    txt = txt.replace('d="M 885 480 L 885 508" stroke="#38bdf8" stroke-width="2"', 'd="M 885 480 L 885 508" stroke="#38bdf8" stroke-width="2" class="flow-cyan"')
    txt = txt.replace('d="M 315 678 L 315 705" stroke="#f59e0b" stroke-width="2"', 'd="M 315 678 L 315 705" stroke="#f59e0b" stroke-width="2" class="flow-amber"')
    txt = txt.replace('d="M 885 678 L 885 705" stroke="#a855f7" stroke-width="2"', 'd="M 885 678 L 885 705" stroke="#a855f7" stroke-width="2" class="flow-purple"')

    file_path.write_text(txt, encoding="utf-8")
    ET.parse(file_path)
    print(f"{file_path}: Successfully animated and validated as 100% valid XML!")
