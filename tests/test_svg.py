import io
import pygame

pygame.init()

svg_solid = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
<rect x="10" y="10" width="80" height="80" rx="10" fill="#F7F6F2" stroke="#26282B" stroke-width="4"/>
</svg>"""

svg_grad = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
<defs>
  <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#FFFFFF"/>
    <stop offset="100%" stop-color="#E0DED7"/>
  </linearGradient>
</defs>
<rect x="10" y="10" width="80" height="80" rx="10" fill="url(#g)" stroke="#26282B" stroke-width="4"/>
</svg>"""

print("Solid fill test:")
s1 = pygame.image.load(io.BytesIO(svg_solid.encode()))
print("  Loaded:", s1.get_size(), "pixel at (50,50):", s1.get_at((50, 50)))

print("Gradient fill test:")
try:
    s2 = pygame.image.load(io.BytesIO(svg_grad.encode()))
    print("  Loaded:", s2.get_size(), "pixel at (50,50):", s2.get_at((50, 50)))
except Exception as e:
    print("  Failed:", e)
