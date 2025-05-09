def route_cipher(text: str, h: int, step=(1, 2)) -> str:
    """
    Scrive text in una griglia h  w (wrap se serve) e lo rilegge
    seguendo il vettore step sul toro (mod h, mod w).
    """
    clean = ''.join(c for c in text.upper() if c.isalpha())
    w = -(-len(clean) // h)           # ceil division
    grid = list(clean.ljust(h * w))   # padding con spazi
    r = c = 0
    dr, dc = step
    out = []
    for _ in range(h * w):
        out.append(grid[r * w + c])
        r = (r + dr) % h
        c = (c + dc) % w
    return ''.join(out).rstrip()
