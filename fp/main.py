import argparse
from typing import Tuple

from fp.core.run import run, run_time_based

def parse_coords(s: str) -> Tuple[float, float]:
    lat_str, lon_str = s.split(",")
    return float(lat_str), float(lon_str)

def build_parser():
    p = argparse.ArgumentParser(description="Flood prediction pipeline runner")
    p.add_argument("--mode", choices=["single", "time-based"], default="single",
                   help="single run or time-based run")
    p.add_argument("--coords", type=str, default="50.0,16.0",
                   help="coordinates as 'lat,lon' (default '50.0,16.0')")
    p.add_argument("--time", type=str, default=None,
                   help="time string for single run (e.g. '2025-06-15')")
    p.add_argument("--interval", type=int, default=3600,
                   help="interval seconds for time-based runs")
    p.add_argument("--out-dir", type=str, default="outputs",
                   help="output directory")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    lat, lon = parse_coords(args.coords)

    if args.mode == "single":
        run(coords={"lat": lat, "lon": lon}, time=args.time, out_dir=args.out_dir)
    else:
        run_time_based(interval=args.interval, coords={"lat": lat, "lon": lon}, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
