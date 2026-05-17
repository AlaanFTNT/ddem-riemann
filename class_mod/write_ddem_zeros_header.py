"""Generate a C header file with the first 200 Riemann zero imaginary parts.
This is included into source/ddem.c at compile time."""
from mpmath import zetazero, mp
from pathlib import Path

mp.dps = 30
N = 200

lines = [
    "/* Auto-generated from write_ddem_zeros_header.py via mpmath at 30-digit precision. */",
    "/* gamma_n = Im(rho_n), first 200 non-trivial zeta zeros, 15 digits each. */",
    "#ifndef DDEM_ZEROS_H",
    "#define DDEM_ZEROS_H",
    "",
    f"#define DDEM_NZEROS {N}",
    "",
    f"static const double ddem_gamma[{N}] = {{",
]
for i, n in enumerate(range(1, N + 1)):
    g = float(zetazero(n).imag)
    sep = "," if i < N - 1 else ""
    lines.append(f"    {g:.15f}{sep}  /* gamma_{n} */")
lines.append("};")
lines.append("")
lines.append("#endif /* DDEM_ZEROS_H */")

out = Path(__file__).parent / "CLASS-source" / "include" / "ddem_zeros.h"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote: {out}")
print(f"First three zeros: {[float(zetazero(n).imag) for n in (1,2,3)]}")
