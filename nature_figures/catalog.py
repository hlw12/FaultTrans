"""Event catalogue: prediction folders, GT sources, display names."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/root/autodl-tmp")
TOE = ROOT / "Test_on_Event"
PGA = ROOT / "PGA_real_eval" / "outputs"
FINDER = ROOT / "CmpFinDer"

EVENTS = {
    "kumamoto": dict(
        label="Kumamoto 2016",
        mw=7.3,
        mech="strike-slip",
        network="KiK-net/K-NET",
        pgv_dir=TOE / "kumamoto",
        pga_dir=PGA / "kumamoto",
        h5=TOE / "kumamoto" / "kumamoto.h5",
        nied_rp=TOE / "kumamoto" / "kumamoto_1.txt",
        nied_cv=TOE / "kumamoto" / "kumamoto_2.txt",
        gt_pref="curved",
        paper_iou=0.439,
        paper_dtheta=18.2,
    ),
    "Miyagi": dict(
        label="Iwate–Miyagi 2008",
        mw=7.2,
        mech="reverse",
        network="KiK-net/K-NET",
        pgv_dir=TOE / "Miyagi",
        pga_dir=PGA / "Miyagi",
        h5=TOE / "Miyagi" / "Miyagi.h5",
        nied_rp=TOE / "Miyagi" / "Miyagi_1.txt",
        nied_cv=None,
        gt_pref="rp",
        paper_iou=0.368,
        paper_dtheta=42.6,
    ),
    "Tottori": dict(
        label="Tottori 2000",
        mw=6.6,
        mech="strike-slip",
        network="KiK-net/K-NET (off-network)",
        pgv_dir=TOE / "Tottori",
        pga_dir=PGA / "Tottori",
        h5=TOE / "Tottori" / "Tottori.h5",
        nied_rp=TOE / "Tottori" / "Tottori_1.txt",
        nied_cv=None,
        gt_pref="rp",
        paper_iou=0.306,
        paper_dtheta=4.6,
    ),
    "ridgecrest": dict(
        label="Ridgecrest 2019",
        mw=7.1,
        mech="strike-slip",
        network="SCSN",
        pgv_dir=TOE / "ridgecrest" / "Ridgecrest",
        pga_dir=PGA / "ridgecrest",
        h5=TOE / "ridgecrest" / "Ridgecrest" / "ridgecrest.h5",
        srcmod={
            "JIN": TOE / "ridgecrest" / "Ridgecrest" / "s2019RIDGEC02JINx.mat",
            "ROSS": TOE / "ridgecrest" / "Ridgecrest" / "s2019RIDGEC02ROSS.mat",
            "XU": TOE / "ridgecrest" / "Ridgecrest" / "s2019RIDGEC02XUxx.mat",
        },
        finder_gt=FINDER / "ridgecrest",
        pku_txt=TOE / "ridgecrest" / "Ridgecrest" / "20190706031953_America_Rupture_Info.txt",
        gt_pref="XU",
        paper_iou=0.521,
        paper_dtheta=12.2,
    ),
    "chichi": dict(
        label="Chi-Chi 1999",
        mw=7.6,
        mech="thrust",
        network="TSMIP",
        pgv_dir=TOE / "chichi" / "Chichi",
        pga_dir=PGA / "chichi",
        h5=TOE / "chichi" / "Chichi" / "Chichi.h5",
        srcmod={
            "CHI": TOE / "chichi" / "Chichi" / "s1999CHICHI01CHIx.mat",
            "MA": TOE / "chichi" / "Chichi" / "s1999CHICHI01MAxx.mat",
            "ZENG": TOE / "chichi" / "Chichi" / "s1999CHICHI01ZENG.mat",
        },
        gt_pref="CHI",
        paper_iou=0.587,
        paper_dtheta=9.6,
    ),
    "Menyuan": dict(
        label="Menyuan 2022",
        mw=6.9,
        mech="strike-slip",
        network="CEA",
        pgv_dir=TOE / "menyuan" / "Menyuan",
        pga_dir=PGA / "Menyuan",
        h5=TOE / "menyuan" / "Menyuan" / "Menyuan.h5",
        pku_txt=TOE / "menyuan" / "Menyuan" / "20220107174530_Menyuan_Rupture_Info.txt",
        gt_pref="pku",
        paper_iou=0.281,
        paper_dtheta=26.5,
    ),
}

PAPER_SIX = ["kumamoto", "Miyagi", "Tottori", "ridgecrest", "chichi", "Menyuan"]
