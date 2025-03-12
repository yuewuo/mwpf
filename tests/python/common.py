import os, math, pytest, sys

""" force import either mwpf or mwpf_rational """
# import mwpf
# import mwpf_rational as mwpf

""" automatic import based on which is available """
if "mwpf" not in globals():
    try:
        import mwpf
    except ImportError as e:
        print(e)
        print("mwpf package not available, trying others...")

if "mwpf" not in globals():
    try:
        import mwpf_rational as mwpf
    except ImportError as e:
        print(e)
        print("mwpf_rational package not available, trying others...")

if "mwpf" not in globals():
    try:
        import mwpf_incr as mwpf
    except ImportError as e:
        print(e)
        print("mwpf_incr package not available, trying others...")

if "mwpf" not in globals():
    try:
        import mwpf_fast as mwpf
    except ImportError as e:
        print(e)
        print("mwpf_fast package not available, trying others...")


def circle_positions(n: int):
    positions = []
    for i in range(n):
        positions.append(
            mwpf.VisualizePosition(
                0.5 + 0.5 * math.cos(2 * math.pi * i / n),
                0.5 + 0.5 * math.sin(2 * math.pi * i / n),
                0,
            )
        )
    return positions
