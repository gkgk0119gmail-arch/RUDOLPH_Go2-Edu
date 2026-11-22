#!/usr/bin/env python3
"""Render a top-down + side preview PNG of a binary PCD (x y z intensity, float32).

usage: python3 tools/render_map_preview.py maps/map_xxx.pcd docs/images/map_xxx.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pcd(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii').strip()
            k, *v = line.split()
            header[k] = v
            if k == 'DATA':
                break
        assert header['DATA'][0] == 'binary', 'only binary PCD supported'
        n = int(header['POINTS'][0])
        nf = len(header['FIELDS'])
        data = np.frombuffer(f.read(n * nf * 4), dtype=np.float32).reshape(n, nf)
    return data[:, :3]


def main(src, dst):
    p = load_pcd(src)
    p = p[np.isfinite(p).all(1)]
    # crop far outliers (glass / open doors) so the room fills the figure
    lo, hi = np.percentile(p[:, :2], [0.5, 99.5], axis=0)
    p = p[((p[:, :2] >= lo - 1) & (p[:, :2] <= hi + 1)).all(1)]
    z_lo, z_hi = np.percentile(p[:, 2], [1, 99])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.4, 1]})
    fig.patch.set_facecolor('#15161a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#15161a')
        ax.tick_params(colors='#aaaaaa')
        for s in ax.spines.values():
            s.set_color('#444444')

    # top-down: drop floor/ceiling so walls & furniture read clearly
    m = (p[:, 2] > z_lo + 0.30) & (p[:, 2] < z_hi - 0.30)
    ax1.scatter(p[m, 0], p[m, 1], c=p[m, 2], s=0.15, cmap='viridis', vmin=z_lo, vmax=z_hi, linewidths=0)
    ax1.scatter([0], [0], c='red', s=60, marker='*', label='start (FAST-LIO origin)')
    ax1.set_aspect('equal')
    ax1.set_title('Top view (floor/ceiling removed, color = height)', color='white')
    ax1.set_xlabel('x [m]', color='#aaaaaa')
    ax1.set_ylabel('y [m]', color='#aaaaaa')
    leg = ax1.legend(loc='upper right', facecolor='#222222')
    for t in leg.get_texts():
        t.set_color('white')

    ax2.scatter(p[:, 0], p[:, 2], c=p[:, 2], s=0.1, cmap='viridis', vmin=z_lo, vmax=z_hi, linewidths=0)
    ax2.set_aspect('equal')
    ax2.set_title('Side view (x-z)', color='white')
    ax2.set_xlabel('x [m]', color='#aaaaaa')
    ax2.set_ylabel('z [m]', color='#aaaaaa')

    fig.suptitle(f'{src.split("/")[-1]}  —  {len(p):,} pts (5 cm voxel)', color='white')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dst, dpi=110, facecolor=fig.get_facecolor())
    print('wrote', dst)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
