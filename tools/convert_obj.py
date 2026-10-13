"""Convert BodyParts3D .obj files (z-up, mm, anterior = -y) into the viewer frame
(y-up, anterior = +z, origin near the heart centre). Usage:
  python tools/convert_obj.py <in.obj> <out.obj>
Drop converted files in frontend/public/models/ and list them in manifest.json under LAD / LCX / RCA."""
import sys
CX, CY, CZ = 20.0, -135.0, 1190.0   # BP3D coordinates of the viewer origin (must match scene.js)

def convert(src, dst):
    out = []
    for line in open(src):
        p = line.split()
        if p and p[0] == "v":
            x, y, z = map(float, p[1:4]); out.append(f"v {x-CX:.3f} {z-CZ:.3f} {-(y-CY):.3f}\n")
        elif p and p[0] == "vn":
            x, y, z = map(float, p[1:4]); out.append(f"vn {x:.5f} {z:.5f} {-y:.5f}\n")
        elif p and (p[0] in ("f", "g", "o")):
            out.append(line)
    open(dst, "w").writelines(out)

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
