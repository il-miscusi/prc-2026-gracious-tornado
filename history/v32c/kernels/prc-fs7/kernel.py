# ==== features.py ====
# Shared feature code for the PRC 2026 taxi-out campaign (from v26's solution.py).
#
# build() makes one row per departure from fields the ranking file publishes; the
# encodings are leave-one-month-out for training rows and fitted on every fit row for
# applied rows.  Kernels embed this file (solutions/core/build_kernel.py).
from __future__ import annotations

import csv
import json
import os
import sys
import time

# %%
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
SEED = 2026
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
VAL_MONTHS = (1, 7)
CLIP_TARGET = 7200.0
TE_CLIP = 3600.0
RARE_MIN = 30
MAX_ROUNDS = int(os.environ.get("COMPBOT_ROUNDS", "5000"))
RUNTIME_CAP_H = 12.0
OUTPUT_DIR = Path.cwd()
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
TZ = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}


def used_h() -> float:
    return (time.time() - T0) / 3600.0


# fs7 options (empty = the fs6 store, unchanged).  build_kernel.py --opts fills them:
#   match: "v7"     one-to-one ADS-B event <-> flight assignment with callsign aliases (IATA -> ICAO),
#                   a 900 s window on takeoff / first airborne point; time-only matches feed own-flight
#                   columns only when the aircraft type agrees and the first airborne point lines up
#                   (match conf 1.5); the rest are demoted to identity-free event-pool columns (evp_*)
#   airline: "icao" airline key from the callsign / mapped IATA designator (not FLIGHT_mvt[:3]);
#                   stand_lb_other, the inbound link and a next-occupant bound use it
#   pru: true       EUROCONTROL PRU published airport-month taxi-out levels (PRU_TAB)
#   demean: true    airport-month-demeaned NM lags (label-free, per frame)
#   ext: "glob"     merge an external per-movement table keyed by MVT_ID_mvt (e.g. opdi_*.parquet)
#   openflights: true  fill IATA designators missing from the data-derived map from OpenFlights
#                   airlines.dat (ODbL 1.0, openflights.org/data), fetched in the kernel
FS_OPTS = {'match': 'v7', 'airline': 'icao', 'pru': True, 'demean': True, 'openflights': True}
PRU_TAB = {'LFPG|2025|1': {'pru_pub': 934.5219, 'pru_ref': 695.0471, 'pru_deice_sh': 0.0906, 'pru_noref_sh': 0.077, 'pru_pub_dev': -33.1952}, 'EDDF|2025|1': {'pru_pub': 831.6665, 'pru_ref': 642.5112, 'pru_deice_sh': 0.0466, 'pru_noref_sh': 0.024, 'pru_pub_dev': -20.1982}, 'EDDM|2025|1': {'pru_pub': 705.8452, 'pru_ref': 558.7263, 'pru_deice_sh': 0.1994, 'pru_noref_sh': 0.0322, 'pru_pub_dev': -65.0766}, 'LIRF|2025|1': {'pru_pub': 1055.8255, 'pru_ref': 750.616, 'pru_deice_sh': 0.0011, 'pru_noref_sh': 0.0632, 'pru_pub_dev': -77.6421}, 'EHAM|2025|1': {'pru_pub': 803.423, 'pru_ref': 545.3423, 'pru_deice_sh': 0.0089, 'pru_noref_sh': 0.0402, 'pru_pub_dev': 21.1044}, 'LEBL|2025|1': {'pru_pub': 885.0293, 'pru_ref': 715.6945, 'pru_deice_sh': 0.0023, 'pru_noref_sh': 0.0298, 'pru_pub_dev': -61.5199}, 'LEMD|2025|1': {'pru_pub': 974.1969, 'pru_ref': 785.5745, 'pru_deice_sh': 0.0206, 'pru_noref_sh': 0.1239, 'pru_pub_dev': -31.5814}, 'LSZH|2025|1': {'pru_pub': 689.1588, 'pru_ref': 508.3978, 'pru_deice_sh': 0.0986, 'pru_noref_sh': 0.0605, 'pru_pub_dev': -23.3563}, 'LTFM|2025|1': {'pru_pub': 1030.6575, 'pru_ref': 767.2741, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0727, 'pru_pub_dev': 14.8729}, 'EGLL|2025|1': {'pru_pub': 1328.2116, 'pru_ref': 974.742, 'pru_deice_sh': 0.0004, 'pru_noref_sh': 0.0135, 'pru_pub_dev': -35.0066}, 'LFPG|2025|2': {'pru_pub': 920.3016, 'pru_ref': 700.3567, 'pru_deice_sh': 0.0572, 'pru_noref_sh': 0.0829, 'pru_pub_dev': -47.4155}, 'EDDF|2025|2': {'pru_pub': 826.4284, 'pru_ref': 640.5936, 'pru_deice_sh': 0.0212, 'pru_noref_sh': 0.0356, 'pru_pub_dev': -25.4363}, 'EDDM|2025|2': {'pru_pub': 719.618, 'pru_ref': 577.2386, 'pru_deice_sh': 0.155, 'pru_noref_sh': 0.0392, 'pru_pub_dev': -51.3039}, 'LIRF|2025|2': {'pru_pub': 1037.5504, 'pru_ref': 728.0546, 'pru_deice_sh': 0.0021, 'pru_noref_sh': 0.0191, 'pru_pub_dev': -95.9172}, 'EHAM|2025|2': {'pru_pub': 813.1343, 'pru_ref': 589.667, 'pru_deice_sh': 0.0119, 'pru_noref_sh': 0.0818, 'pru_pub_dev': 30.8157}, 'LEBL|2025|2': {'pru_pub': 905.7314, 'pru_ref': 716.9857, 'pru_deice_sh': 0.0016, 'pru_noref_sh': 0.0333, 'pru_pub_dev': -40.8179}, 'LEMD|2025|2': {'pru_pub': 964.6365, 'pru_ref': 781.6182, 'pru_deice_sh': 0.0155, 'pru_noref_sh': 0.0872, 'pru_pub_dev': -41.1418}, 'LSZH|2025|2': {'pru_pub': 692.5538, 'pru_ref': 511.5295, 'pru_deice_sh': 0.0632, 'pru_noref_sh': 0.0403, 'pru_pub_dev': -19.9613}, 'LTFM|2025|2': {'pru_pub': 989.5166, 'pru_ref': 725.8099, 'pru_deice_sh': 0.1112, 'pru_noref_sh': 0.0996, 'pru_pub_dev': -26.268}, 'EGLL|2025|2': {'pru_pub': 1321.1061, 'pru_ref': 963.6067, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0182, 'pru_pub_dev': -42.1121}, 'LFPG|2025|3': {'pru_pub': 904.3852, 'pru_ref': 702.7352, 'pru_deice_sh': 0.0145, 'pru_noref_sh': 0.0935, 'pru_pub_dev': -63.332}, 'EDDF|2025|3': {'pru_pub': 822.8577, 'pru_ref': 635.599, 'pru_deice_sh': 0.0051, 'pru_noref_sh': 0.0411, 'pru_pub_dev': -29.007}, 'EDDM|2025|3': {'pru_pub': 736.6727, 'pru_ref': 574.6436, 'pru_deice_sh': 0.0572, 'pru_noref_sh': 0.0294, 'pru_pub_dev': -34.2491}, 'LIRF|2025|3': {'pru_pub': 1078.7835, 'pru_ref': 739.3082, 'pru_deice_sh': 0.0023, 'pru_noref_sh': 0.038, 'pru_pub_dev': -54.684}, 'EHAM|2025|3': {'pru_pub': 846.1531, 'pru_ref': 633.1537, 'pru_deice_sh': 0.0128, 'pru_noref_sh': 0.1171, 'pru_pub_dev': 63.8345}, 'LEBL|2025|3': {'pru_pub': 931.7835, 'pru_ref': 699.5, 'pru_deice_sh': 0.0017, 'pru_noref_sh': 0.0416, 'pru_pub_dev': -14.7657}, 'LEMD|2025|3': {'pru_pub': 1034.9324, 'pru_ref': 788.8493, 'pru_deice_sh': 0.0023, 'pru_noref_sh': 0.1331, 'pru_pub_dev': 29.154}, 'LSZH|2025|3': {'pru_pub': 698.2679, 'pru_ref': 514.1621, 'pru_deice_sh': 0.0338, 'pru_noref_sh': 0.1107, 'pru_pub_dev': -14.2472}, 'LTFM|2025|3': {'pru_pub': 1000.0836, 'pru_ref': 749.7472, 'pru_deice_sh': 0.0019, 'pru_noref_sh': 0.104, 'pru_pub_dev': -15.701}, 'EGLL|2025|3': {'pru_pub': 1357.3335, 'pru_ref': 966.8929, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0174, 'pru_pub_dev': -5.8848}, 'LFPG|2025|4': {'pru_pub': 950.3717, 'pru_ref': 705.5736, 'pru_deice_sh': 0.0005, 'pru_noref_sh': 0.0833, 'pru_pub_dev': -17.3455}, 'EDDF|2025|4': {'pru_pub': 838.1857, 'pru_ref': 638.2363, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0331, 'pru_pub_dev': -13.679}, 'EDDM|2025|4': {'pru_pub': 767.5777, 'pru_ref': 586.1799, 'pru_deice_sh': 0.0138, 'pru_noref_sh': 0.0311, 'pru_pub_dev': -3.3442}, 'LIRF|2025|4': {'pru_pub': 1147.2199, 'pru_ref': 742.2388, 'pru_deice_sh': 0.002, 'pru_noref_sh': 0.0246, 'pru_pub_dev': 13.7523}, 'EHAM|2025|4': {'pru_pub': 844.546, 'pru_ref': 662.705, 'pru_deice_sh': 0.012, 'pru_noref_sh': 0.088, 'pru_pub_dev': 62.2275}, 'LEBL|2025|4': {'pru_pub': 949.5935, 'pru_ref': 719.7461, 'pru_deice_sh': 0.0025, 'pru_noref_sh': 0.0352, 'pru_pub_dev': 3.0442}, 'LEMD|2025|4': {'pru_pub': 996.3496, 'pru_ref': 792.7281, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.133, 'pru_pub_dev': -9.4288}, 'LSZH|2025|4': {'pru_pub': 716.928, 'pru_ref': 522.0061, 'pru_deice_sh': 0.0081, 'pru_noref_sh': 0.1053, 'pru_pub_dev': 4.413}, 'LTFM|2025|4': {'pru_pub': 1005.1681, 'pru_ref': 711.1399, 'pru_deice_sh': 0.0005, 'pru_noref_sh': 0.0981, 'pru_pub_dev': -10.6165}, 'EGLL|2025|4': {'pru_pub': 1310.1445, 'pru_ref': 960.3976, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0078, 'pru_pub_dev': -53.0737}, 'LFPG|2025|5': {'pru_pub': 977.748, 'pru_ref': 700.0689, 'pru_deice_sh': 0.0005, 'pru_noref_sh': 0.0765, 'pru_pub_dev': 10.0309}, 'EDDF|2025|5': {'pru_pub': 855.2846, 'pru_ref': 632.2436, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0304, 'pru_pub_dev': 3.4198}, 'EDDM|2025|5': {'pru_pub': 768.1263, 'pru_ref': 565.5833, 'pru_deice_sh': 0.0084, 'pru_noref_sh': 0.0315, 'pru_pub_dev': -2.7955}, 'LIRF|2025|5': {'pru_pub': 1154.9653, 'pru_ref': 739.8364, 'pru_deice_sh': 0.0023, 'pru_noref_sh': 0.018, 'pru_pub_dev': 21.4977}, 'EHAM|2025|5': {'pru_pub': 832.296, 'pru_ref': 657.4025, 'pru_deice_sh': 0.0105, 'pru_noref_sh': 0.0678, 'pru_pub_dev': 49.9775}, 'LEBL|2025|5': {'pru_pub': 964.8745, 'pru_ref': 726.5255, 'pru_deice_sh': 0.0027, 'pru_noref_sh': 0.0362, 'pru_pub_dev': 18.3253}, 'LEMD|2025|5': {'pru_pub': 991.9842, 'pru_ref': 787.5729, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1178, 'pru_pub_dev': -13.7941}, 'LSZH|2025|5': {'pru_pub': 721.908, 'pru_ref': 518.2496, 'pru_deice_sh': 0.0041, 'pru_noref_sh': 0.0922, 'pru_pub_dev': 9.3929}, 'LTFM|2025|5': {'pru_pub': 1038.7957, 'pru_ref': 754.9714, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1439, 'pru_pub_dev': 23.0111}, 'EGLL|2025|5': {'pru_pub': 1350.2277, 'pru_ref': 962.4854, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0073, 'pru_pub_dev': -12.9905}, 'LFPG|2025|6': {'pru_pub': 928.1251, 'pru_ref': 695.1861, 'pru_deice_sh': 0.001, 'pru_noref_sh': 0.0753, 'pru_pub_dev': -39.592}, 'EDDF|2025|6': {'pru_pub': 864.0774, 'pru_ref': 642.4572, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0247, 'pru_pub_dev': 12.2126}, 'EDDM|2025|6': {'pru_pub': 788.9038, 'pru_ref': 565.9325, 'pru_deice_sh': 0.0077, 'pru_noref_sh': 0.0381, 'pru_pub_dev': 17.9819}, 'LIRF|2025|6': {'pru_pub': 1207.2256, 'pru_ref': 738.5718, 'pru_deice_sh': 0.0041, 'pru_noref_sh': 0.0211, 'pru_pub_dev': 73.7581}, 'EHAM|2025|6': {'pru_pub': 714.3552, 'pru_ref': 544.1304, 'pru_deice_sh': 0.0129, 'pru_noref_sh': 0.0424, 'pru_pub_dev': -67.9634}, 'LEBL|2025|6': {'pru_pub': 968.929, 'pru_ref': 729.8912, 'pru_deice_sh': 0.0028, 'pru_noref_sh': 0.0467, 'pru_pub_dev': 22.3798}, 'LEMD|2025|6': {'pru_pub': 1020.7785, 'pru_ref': 785.4551, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1152, 'pru_pub_dev': 15.0002}, 'LSZH|2025|6': {'pru_pub': 729.9177, 'pru_ref': 514.738, 'pru_deice_sh': 0.0095, 'pru_noref_sh': 0.079, 'pru_pub_dev': 17.4026}, 'LTFM|2025|6': {'pru_pub': 982.2325, 'pru_ref': 694.4896, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0729, 'pru_pub_dev': -33.5521}, 'EGLL|2025|6': {'pru_pub': 1382.813, 'pru_ref': 978.6848, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0091, 'pru_pub_dev': 19.5948}, 'LFPG|2025|7': {'pru_pub': 982.7939, 'pru_ref': 697.5133, 'pru_deice_sh': 0.0007, 'pru_noref_sh': 0.2439, 'pru_pub_dev': 15.0768}, 'EDDF|2025|7': {'pru_pub': 866.4698, 'pru_ref': 641.7677, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.022, 'pru_pub_dev': 14.605}, 'EDDM|2025|7': {'pru_pub': 804.9772, 'pru_ref': 560.1434, 'pru_deice_sh': 0.007, 'pru_noref_sh': 0.0309, 'pru_pub_dev': 34.0553}, 'LIRF|2025|7': {'pru_pub': 1284.1066, 'pru_ref': 740.8662, 'pru_deice_sh': 0.0066, 'pru_noref_sh': 0.0234, 'pru_pub_dev': 150.639}, 'EHAM|2025|7': {'pru_pub': 785.7588, 'pru_ref': 603.3261, 'pru_deice_sh': 0.01, 'pru_noref_sh': 0.0488, 'pru_pub_dev': 3.4403}, 'LEBL|2025|7': {'pru_pub': 998.6247, 'pru_ref': 727.637, 'pru_deice_sh': 0.0028, 'pru_noref_sh': 0.0504, 'pru_pub_dev': 52.0754}, 'LEMD|2025|7': {'pru_pub': 1027.7966, 'pru_ref': 783.8516, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1207, 'pru_pub_dev': 22.0183}, 'LSZH|2025|7': {'pru_pub': 732.3413, 'pru_ref': 518.4107, 'pru_deice_sh': 0.0067, 'pru_noref_sh': 0.0718, 'pru_pub_dev': 19.8262}, 'LTFM|2025|7': {'pru_pub': 1070.6574, 'pru_ref': 700.0328, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0897, 'pru_pub_dev': 54.8728}, 'EGLL|2025|7': {'pru_pub': 1427.4794, 'pru_ref': 982.8365, 'pru_deice_sh': 0.0007, 'pru_noref_sh': 0.0082, 'pru_pub_dev': 64.2611}, 'LFPG|2025|8': {'pru_pub': 1011.7968, 'pru_ref': 702.248, 'pru_deice_sh': 0.0006, 'pru_noref_sh': 0.3037, 'pru_pub_dev': 44.0796}, 'EDDF|2025|8': {'pru_pub': 850.7751, 'pru_ref': 643.3911, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0218, 'pru_pub_dev': -1.0897}, 'EDDM|2025|8': {'pru_pub': 792.3818, 'pru_ref': 570.8883, 'pru_deice_sh': 0.0083, 'pru_noref_sh': 0.0343, 'pru_pub_dev': 21.46}, 'LIRF|2025|8': {'pru_pub': 1198.4937, 'pru_ref': 739.4479, 'pru_deice_sh': 0.0044, 'pru_noref_sh': 0.0115, 'pru_pub_dev': 65.0261}, 'EHAM|2025|8': {'pru_pub': 764.8698, 'pru_ref': 589.8035, 'pru_deice_sh': 0.0094, 'pru_noref_sh': 0.0373, 'pru_pub_dev': -17.4488}, 'LEBL|2025|8': {'pru_pub': 971.3305, 'pru_ref': 726.6311, 'pru_deice_sh': 0.0034, 'pru_noref_sh': 0.045, 'pru_pub_dev': 24.7812}, 'LEMD|2025|8': {'pru_pub': 1035.5663, 'pru_ref': 787.0883, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0997, 'pru_pub_dev': 29.7879}, 'LSZH|2025|8': {'pru_pub': 727.3802, 'pru_ref': 518.4332, 'pru_deice_sh': 0.0062, 'pru_noref_sh': 0.0636, 'pru_pub_dev': 14.8651}, 'LTFM|2025|8': {'pru_pub': 1003.3689, 'pru_ref': 703.0844, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0722, 'pru_pub_dev': -12.4156}, 'EGLL|2025|8': {'pru_pub': 1367.524, 'pru_ref': 975.4326, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.0097, 'pru_pub_dev': 4.3058}, 'LFPG|2025|9': {'pru_pub': 1011.8478, 'pru_ref': 718.7868, 'pru_deice_sh': 0.0006, 'pru_noref_sh': 0.2282, 'pru_pub_dev': 44.1307}, 'EDDF|2025|9': {'pru_pub': 869.4372, 'pru_ref': 636.6263, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0318, 'pru_pub_dev': 17.5725}, 'EDDM|2025|9': {'pru_pub': 810.2887, 'pru_ref': 577.2333, 'pru_deice_sh': 0.0067, 'pru_noref_sh': 0.0474, 'pru_pub_dev': 39.3668}, 'LIRF|2025|9': {'pru_pub': 1169.2706, 'pru_ref': 738.3726, 'pru_deice_sh': 0.0029, 'pru_noref_sh': 0.01, 'pru_pub_dev': 35.803}, 'EHAM|2025|9': {'pru_pub': 728.309, 'pru_ref': 548.1118, 'pru_deice_sh': 0.0086, 'pru_noref_sh': 0.0477, 'pru_pub_dev': -54.0096}, 'LEBL|2025|9': {'pru_pub': 978.4044, 'pru_ref': 718.2321, 'pru_deice_sh': 0.0027, 'pru_noref_sh': 0.0481, 'pru_pub_dev': 31.8552}, 'LEMD|2025|9': {'pru_pub': 1034.7896, 'pru_ref': 796.5563, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1085, 'pru_pub_dev': 29.0113}, 'LSZH|2025|9': {'pru_pub': 717.8922, 'pru_ref': 515.4214, 'pru_deice_sh': 0.0037, 'pru_noref_sh': 0.0681, 'pru_pub_dev': 5.3771}, 'LTFM|2025|9': {'pru_pub': 976.6537, 'pru_ref': 696.226, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0544, 'pru_pub_dev': -39.1309}, 'EGLL|2025|9': {'pru_pub': 1408.2531, 'pru_ref': 977.0606, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.008, 'pru_pub_dev': 45.0349}, 'LFPG|2025|10': {'pru_pub': 1011.3227, 'pru_ref': 726.8016, 'pru_deice_sh': 0.0006, 'pru_noref_sh': 0.2017, 'pru_pub_dev': 43.6055}, 'EDDF|2025|10': {'pru_pub': 863.488, 'pru_ref': 642.7818, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0262, 'pru_pub_dev': 11.6233}, 'EDDM|2025|10': {'pru_pub': 799.5883, 'pru_ref': 579.486, 'pru_deice_sh': 0.0121, 'pru_noref_sh': 0.0497, 'pru_pub_dev': 28.6665}, 'LIRF|2025|10': {'pru_pub': 1164.9454, 'pru_ref': 741.9316, 'pru_deice_sh': 0.0027, 'pru_noref_sh': 0.0316, 'pru_pub_dev': 31.4779}, 'EHAM|2025|10': {'pru_pub': 710.7949, 'pru_ref': 522.8186, 'pru_deice_sh': 0.0098, 'pru_noref_sh': 0.1302, 'pru_pub_dev': -71.5236}, 'LEBL|2025|10': {'pru_pub': 974.2776, 'pru_ref': 728.6068, 'pru_deice_sh': 0.0014, 'pru_noref_sh': 0.0414, 'pru_pub_dev': 27.7283}, 'LEMD|2025|10': {'pru_pub': 1022.8157, 'pru_ref': 799.4336, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.1239, 'pru_pub_dev': 17.0374}, 'LSZH|2025|10': {'pru_pub': 709.0812, 'pru_ref': 517.5085, 'pru_deice_sh': 0.0048, 'pru_noref_sh': 0.0809, 'pru_pub_dev': -3.4338}, 'LTFM|2025|10': {'pru_pub': 1062.0209, 'pru_ref': 758.2075, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0983, 'pru_pub_dev': 46.2363}, 'EGLL|2025|10': {'pru_pub': 1375.0578, 'pru_ref': 980.6755, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.0068, 'pru_pub_dev': 11.8395}, 'LFPG|2025|11': {'pru_pub': 1020.4676, 'pru_ref': 733.6499, 'pru_deice_sh': 0.0318, 'pru_noref_sh': 0.1474, 'pru_pub_dev': 52.7505}, 'EDDF|2025|11': {'pru_pub': 866.2091, 'pru_ref': 647.7332, 'pru_deice_sh': 0.0221, 'pru_noref_sh': 0.023, 'pru_pub_dev': 14.3443}, 'EDDM|2025|11': {'pru_pub': 764.8684, 'pru_ref': 573.2078, 'pru_deice_sh': 0.1023, 'pru_noref_sh': 0.0329, 'pru_pub_dev': -6.0534}, 'LIRF|2025|11': {'pru_pub': 1036.0551, 'pru_ref': 730.8469, 'pru_deice_sh': 0.0017, 'pru_noref_sh': 0.057, 'pru_pub_dev': -97.4124}, 'EHAM|2025|11': {'pru_pub': 763.3345, 'pru_ref': 546.2966, 'pru_deice_sh': 0.0087, 'pru_noref_sh': 0.0583, 'pru_pub_dev': -18.984}, 'LEBL|2025|11': {'pru_pub': 907.0992, 'pru_ref': 723.9294, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0405, 'pru_pub_dev': -39.45}, 'LEMD|2025|11': {'pru_pub': 988.1374, 'pru_ref': 795.3054, 'pru_deice_sh': 0.008, 'pru_noref_sh': 0.1414, 'pru_pub_dev': -17.6409}, 'LSZH|2025|11': {'pru_pub': 697.8148, 'pru_ref': 513.5992, 'pru_deice_sh': 0.0645, 'pru_noref_sh': 0.0466, 'pru_pub_dev': -14.7003}, 'LTFM|2025|11': {'pru_pub': 1050.92, 'pru_ref': 781.0555, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.1208, 'pru_pub_dev': 35.1354}, 'EGLL|2025|11': {'pru_pub': 1355.7461, 'pru_ref': 982.1257, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0094, 'pru_pub_dev': -7.4721}, 'LFPG|2025|12': {'pru_pub': 958.9232, 'pru_ref': 719.8684, 'pru_deice_sh': 0.0465, 'pru_noref_sh': 0.1118, 'pru_pub_dev': -8.7939}, 'EDDF|2025|12': {'pru_pub': 867.4975, 'pru_ref': 650.2767, 'pru_deice_sh': 0.0193, 'pru_noref_sh': 0.0219, 'pru_pub_dev': 15.6328}, 'EDDM|2025|12': {'pru_pub': 792.2142, 'pru_ref': 582.7435, 'pru_deice_sh': 0.1794, 'pru_noref_sh': 0.0288, 'pru_pub_dev': 21.2924}, 'LIRF|2025|12': {'pru_pub': 1067.1693, 'pru_ref': 728.8675, 'pru_deice_sh': 0.0026, 'pru_noref_sh': 0.0423, 'pru_pub_dev': -66.2983}, 'EHAM|2025|12': {'pru_pub': 780.8481, 'pru_ref': 581.7975, 'pru_deice_sh': 0.0102, 'pru_noref_sh': 0.0585, 'pru_pub_dev': -1.4705}, 'LEBL|2025|12': {'pru_pub': 922.9134, 'pru_ref': 712.9075, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0426, 'pru_pub_dev': -23.6358}, 'LEMD|2025|12': {'pru_pub': 977.3561, 'pru_ref': 790.0124, 'pru_deice_sh': 0.0181, 'pru_noref_sh': 0.0859, 'pru_pub_dev': -28.4222}, 'LSZH|2025|12': {'pru_pub': 716.9371, 'pru_ref': 520.8432, 'pru_deice_sh': 0.0425, 'pru_noref_sh': 0.0668, 'pru_pub_dev': 4.4221}, 'LTFM|2025|12': {'pru_pub': 979.3401, 'pru_ref': 715.5607, 'pru_deice_sh': 0.0005, 'pru_noref_sh': 0.0655, 'pru_pub_dev': -36.4445}, 'EGLL|2025|12': {'pru_pub': 1374.7219, 'pru_ref': 977.6185, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0075, 'pru_pub_dev': 11.5037}, 'LFPG|2026|1': {'pru_pub': 928.1846, 'pru_ref': 704.1896, 'pru_deice_sh': 0.0977, 'pru_noref_sh': 0.1098, 'pru_pub_dev': -39.5325}, 'EDDF|2026|1': {'pru_pub': 853.5087, 'pru_ref': 654.3636, 'pru_deice_sh': 0.1272, 'pru_noref_sh': 0.0244, 'pru_pub_dev': 1.6439}, 'EDDM|2026|1': {'pru_pub': 754.9653, 'pru_ref': 582.367, 'pru_deice_sh': 0.3105, 'pru_noref_sh': 0.0241, 'pru_pub_dev': -15.9566}, 'LIRF|2026|1': {'pru_pub': 1043.263, 'pru_ref': 723.9286, 'pru_deice_sh': 0.004, 'pru_noref_sh': 0.0318, 'pru_pub_dev': -90.2046}, 'EHAM|2026|1': {'pru_pub': 982.0797, 'pru_ref': 560.2431, 'pru_deice_sh': 0.0107, 'pru_noref_sh': 0.1255, 'pru_pub_dev': 199.7611}, 'LEBL|2026|1': {'pru_pub': 883.0527, 'pru_ref': 708.8295, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0484, 'pru_pub_dev': -63.4965}, 'LEMD|2026|1': {'pru_pub': 987.2564, 'pru_ref': 792.3358, 'pru_deice_sh': 0.0312, 'pru_noref_sh': 0.1156, 'pru_pub_dev': -18.5219}, 'LSZH|2026|1': {'pru_pub': 710.9603, 'pru_ref': 515.9726, 'pru_deice_sh': 0.1971, 'pru_noref_sh': 0.0581, 'pru_pub_dev': -1.5548}, 'LTFM|2026|1': {'pru_pub': 1088.6914, 'pru_ref': 795.0885, 'pru_deice_sh': 0.0485, 'pru_noref_sh': 0.1177, 'pru_pub_dev': 72.9068}, 'EGLL|2026|1': {'pru_pub': 1334.8543, 'pru_ref': 974.3368, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0066, 'pru_pub_dev': -28.3639}, 'LFPG|2026|2': {'pru_pub': 918.1899, 'pru_ref': 705.6351, 'pru_deice_sh': 0.0087, 'pru_noref_sh': 0.1305, 'pru_pub_dev': -49.5272}, 'EDDF|2026|2': {'pru_pub': 857.2418, 'pru_ref': 645.1321, 'pru_deice_sh': 0.0239, 'pru_noref_sh': 0.0246, 'pru_pub_dev': 5.3771}, 'EDDM|2026|2': {'pru_pub': 752.5255, 'pru_ref': 582.0122, 'pru_deice_sh': 0.1301, 'pru_noref_sh': 0.0397, 'pru_pub_dev': -18.3964}, 'LIRF|2026|2': {'pru_pub': 1033.6973, 'pru_ref': 718.0789, 'pru_deice_sh': 0.0021, 'pru_noref_sh': 0.0128, 'pru_pub_dev': -99.7702}, 'EHAM|2026|2': {'pru_pub': 781.5471, 'pru_ref': 563.7234, 'pru_deice_sh': 0.0096, 'pru_noref_sh': 0.0906, 'pru_pub_dev': -0.7715}, 'LEBL|2026|2': {'pru_pub': 918.9948, 'pru_ref': 729.1385, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.0321, 'pru_pub_dev': -27.5544}, 'LEMD|2026|2': {'pru_pub': 1010.8393, 'pru_ref': 796.9438, 'pru_deice_sh': 0.0022, 'pru_noref_sh': 0.1662, 'pru_pub_dev': 5.061}, 'LSZH|2026|2': {'pru_pub': 705.8147, 'pru_ref': 523.794, 'pru_deice_sh': 0.0553, 'pru_noref_sh': 0.0549, 'pru_pub_dev': -6.7004}, 'LTFM|2026|2': {'pru_pub': 1045.0303, 'pru_ref': 772.2298, 'pru_deice_sh': 0.0102, 'pru_noref_sh': 0.1216, 'pru_pub_dev': 29.2457}, 'EGLL|2026|2': {'pru_pub': 1318.9044, 'pru_ref': 977.76, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.0108, 'pru_pub_dev': -44.3138}, 'LFPG|2026|3': {'pru_pub': 909.2077, 'pru_ref': 706.3916, 'pru_deice_sh': 0.0037, 'pru_noref_sh': 0.1187, 'pru_pub_dev': -58.5094}, 'EDDF|2026|3': {'pru_pub': 871.611, 'pru_ref': 662.3385, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.145, 'pru_pub_dev': 19.7463}, 'EDDM|2026|3': {'pru_pub': 744.5445, 'pru_ref': 585.1043, 'pru_deice_sh': 0.0584, 'pru_noref_sh': 0.0379, 'pru_pub_dev': -26.3774}, 'LIRF|2026|3': {'pru_pub': 1114.2555, 'pru_ref': 720.9085, 'pru_deice_sh': 0.0058, 'pru_noref_sh': 0.107, 'pru_pub_dev': -19.212}, 'EHAM|2026|3': {'pru_pub': 781.4815, 'pru_ref': 594.2367, 'pru_deice_sh': 0.0118, 'pru_noref_sh': 0.0388, 'pru_pub_dev': -0.837}, 'LEBL|2026|3': {'pru_pub': 933.7865, 'pru_ref': 707.9175, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0462, 'pru_pub_dev': -12.7627}, 'LEMD|2026|3': {'pru_pub': 1026.4368, 'pru_ref': 794.5504, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0831, 'pru_pub_dev': 20.6585}, 'LSZH|2026|3': {'pru_pub': 700.0405, 'pru_ref': 520.6949, 'pru_deice_sh': 0.0508, 'pru_noref_sh': 0.0754, 'pru_pub_dev': -12.4746}, 'LTFM|2026|3': {'pru_pub': 957.5273, 'pru_ref': 720.7191, 'pru_deice_sh': 0.0007, 'pru_noref_sh': 0.097, 'pru_pub_dev': -58.2573}, 'EGLL|2026|3': {'pru_pub': 1274.8384, 'pru_ref': 975.3509, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0178, 'pru_pub_dev': -88.3799}, 'LFPG|2026|4': {'pru_pub': 913.1237, 'pru_ref': 713.3205, 'pru_deice_sh': 0.0009, 'pru_noref_sh': 0.104, 'pru_pub_dev': -54.5935}, 'EDDF|2026|4': {'pru_pub': 824.4117, 'pru_ref': 642.5792, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.0338, 'pru_pub_dev': -27.4531}, 'EDDM|2026|4': {'pru_pub': 753.4586, 'pru_ref': 593.2081, 'pru_deice_sh': 0.0183, 'pru_noref_sh': 0.0592, 'pru_pub_dev': -17.4633}, 'LIRF|2026|4': {'pru_pub': 1165.786, 'pru_ref': 733.6961, 'pru_deice_sh': 0.0041, 'pru_noref_sh': 0.0854, 'pru_pub_dev': 32.3184}, 'EHAM|2026|4': {'pru_pub': 788.7787, 'pru_ref': 615.7954, 'pru_deice_sh': 0.0101, 'pru_noref_sh': 0.0657, 'pru_pub_dev': 6.4601}, 'LEBL|2026|4': {'pru_pub': 923.9762, 'pru_ref': 720.7429, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0388, 'pru_pub_dev': -22.573}, 'LEMD|2026|4': {'pru_pub': 1014.7137, 'pru_ref': 795.4043, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1088, 'pru_pub_dev': 8.9354}, 'LSZH|2026|4': {'pru_pub': 724.9717, 'pru_ref': 524.9124, 'pru_deice_sh': 0.0123, 'pru_noref_sh': 0.1084, 'pru_pub_dev': 12.4566}, 'LTFM|2026|4': {'pru_pub': 968.5627, 'pru_ref': 727.2768, 'pru_deice_sh': 0.0003, 'pru_noref_sh': 0.0944, 'pru_pub_dev': -47.2219}, 'EGLL|2026|4': {'pru_pub': 1245.3023, 'pru_ref': 963.1836, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.0097, 'pru_pub_dev': -117.916}, 'LFPG|2026|5': {'pru_pub': 929.6365, 'pru_ref': 708.527, 'pru_deice_sh': 0.0002, 'pru_noref_sh': 0.11, 'pru_pub_dev': -38.0807}, 'EDDF|2026|5': {'pru_pub': 841.9033, 'pru_ref': 640.6235, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0702, 'pru_pub_dev': -9.9615}, 'EDDM|2026|5': {'pru_pub': 772.4403, 'pru_ref': 586.3151, 'pru_deice_sh': 0.0085, 'pru_noref_sh': 0.0575, 'pru_pub_dev': 1.5185}, 'LIRF|2026|5': {'pru_pub': 1173.1703, 'pru_ref': 726.363, 'pru_deice_sh': 0.0025, 'pru_noref_sh': 0.035, 'pru_pub_dev': 39.7028}, 'EHAM|2026|5': {'pru_pub': 753.8026, 'pru_ref': 580.5447, 'pru_deice_sh': 0.0094, 'pru_noref_sh': 0.0813, 'pru_pub_dev': -28.516}, 'LEBL|2026|5': {'pru_pub': 965.2489, 'pru_ref': 716.5369, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0475, 'pru_pub_dev': 18.6997}, 'LEMD|2026|5': {'pru_pub': 1019.4188, 'pru_ref': 799.1162, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1392, 'pru_pub_dev': 13.6405}, 'LSZH|2026|5': {'pru_pub': 728.3972, 'pru_ref': 523.8652, 'pru_deice_sh': 0.0046, 'pru_noref_sh': 0.0587, 'pru_pub_dev': 15.8821}, 'LTFM|2026|5': {'pru_pub': 983.0325, 'pru_ref': 739.6188, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0976, 'pru_pub_dev': -32.7521}, 'EGLL|2026|5': {'pru_pub': 1340.4254, 'pru_ref': 969.5935, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0091, 'pru_pub_dev': -22.7928}, 'LFPG|2026|6': {'pru_pub': 963.6536, 'pru_ref': 711.6608, 'pru_deice_sh': 0.0007, 'pru_noref_sh': 0.1223, 'pru_pub_dev': -4.0636}, 'EDDF|2026|6': {'pru_pub': 867.3474, 'pru_ref': 658.813, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0826, 'pru_pub_dev': 15.4826}, 'EDDM|2026|6': {'pru_pub': 776.6482, 'pru_ref': 586.4149, 'pru_deice_sh': 0.0095, 'pru_noref_sh': 0.0426, 'pru_pub_dev': 5.7264}, 'LIRF|2026|6': {'pru_pub': 1201.1317, 'pru_ref': 723.7259, 'pru_deice_sh': 0.0028, 'pru_noref_sh': 0.024, 'pru_pub_dev': 67.6641}, 'EHAM|2026|6': {'pru_pub': 742.94, 'pru_ref': 564.5258, 'pru_deice_sh': 0.0096, 'pru_noref_sh': 0.0344, 'pru_pub_dev': -39.3786}, 'LEBL|2026|6': {'pru_pub': 973.3296, 'pru_ref': 724.5679, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0454, 'pru_pub_dev': 26.7804}, 'LEMD|2026|6': {'pru_pub': 1045.6535, 'pru_ref': 805.2756, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.1455, 'pru_pub_dev': 39.8752}, 'LSZH|2026|6': {'pru_pub': 728.0269, 'pru_ref': 524.4617, 'pru_deice_sh': 0.0051, 'pru_noref_sh': 0.0795, 'pru_pub_dev': 15.5118}, 'LTFM|2026|6': {'pru_pub': 989.7956, 'pru_ref': 717.7474, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0649, 'pru_pub_dev': -25.9889}, 'EGLL|2026|6': {'pru_pub': 1367.0242, 'pru_ref': 976.232, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0062, 'pru_pub_dev': 3.806}, 'LFPG|2026|7': {'pru_pub': 965.104, 'pru_ref': 721.3365, 'pru_deice_sh': 0.0005, 'pru_noref_sh': 0.1181, 'pru_pub_dev': -2.6131}, 'EDDF|2026|7': {'pru_pub': 854.2468, 'pru_ref': 662.7075, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0787, 'pru_pub_dev': 2.3821}, 'EDDM|2026|7': {'pru_pub': 776.3288, 'pru_ref': 572.2536, 'pru_deice_sh': 0.0081, 'pru_noref_sh': 0.0328, 'pru_pub_dev': 5.4069}, 'LIRF|2026|7': {'pru_pub': 1214.0387, 'pru_ref': 727.0376, 'pru_deice_sh': 0.0047, 'pru_noref_sh': 0.0611, 'pru_pub_dev': 80.5711}, 'EHAM|2026|7': {'pru_pub': 840.8705, 'pru_ref': 654.1087, 'pru_deice_sh': 0.0087, 'pru_noref_sh': 0.0553, 'pru_pub_dev': 58.5519}, 'LEBL|2026|7': {'pru_pub': 967.2186, 'pru_ref': 710.5441, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.0498, 'pru_pub_dev': 20.6694}, 'LEMD|2026|7': {'pru_pub': 1020.4391, 'pru_ref': 795.7964, 'pru_deice_sh': 0.0, 'pru_noref_sh': 0.147, 'pru_pub_dev': 14.6608}, 'LSZH|2026|7': {'pru_pub': 741.5026, 'pru_ref': 529.9347, 'pru_deice_sh': 0.0065, 'pru_noref_sh': 0.0727, 'pru_pub_dev': 28.9875}, 'LTFM|2026|7': {'pru_pub': 1008.8309, 'pru_ref': 706.8706, 'pru_deice_sh': 0.0001, 'pru_noref_sh': 0.0476, 'pru_pub_dev': -6.9537}, 'EGLL|2026|7': {'pru_pub': 1344.1112, 'pru_ref': 962.0, 'pru_deice_sh': 0.0003, 'pru_noref_sh': 0.0128, 'pru_pub_dev': -19.107}}  #: {"APT|YYYY|M": {pru_pub, pru_ref, pru_deice_sh, pru_noref_sh, pru_pub_dev}}
IATA_MAP: dict = {}


def budget(stage: str, projected_stage_h: float) -> bool:
    projected = used_h() + projected_stage_h
    print(f"RUNTIME_BUDGET: stage={stage} used_h={used_h():.3f} projected_h={projected:.3f}", flush=True)
    return projected < RUNTIME_CAP_H * 0.85


def resolve_data_dir() -> Path:
    explicit = os.environ.get("DATA_ROOT")
    if explicit:
        # An explicit root is never silently replaced: a missing smoke slice
        # once fell through to the full data and trained on it locally.
        hits = [h.parent for h in sorted(Path(explicit).rglob("ranking.parquet"))
                if list(h.parent.glob("training_*.parquet"))]
        if not hits:
            raise FileNotFoundError(f"DATA_ROOT={explicit} holds no ranking.parquet beside training files")
        return hits[0]
    roots = []
    roots += [
        Path("/kaggle/input"),
        Path.cwd(),
        # <redacted: development machine data path>
    ]
    for root in roots:
        if root.exists():
            for hit in sorted(root.rglob("ranking.parquet")):
                if list(hit.parent.glob("training_*.parquet")):
                    return hit.parent
    raise FileNotFoundError("no directory holds ranking.parquet beside training_*.parquet")


# %%
# Event-window helpers.  Every count is over one key (airport or airport|runway)
# and uses searchsorted on that key's sorted event times.
def _by_key(q_key, r_key):
    q_codes, uniq = pd.factorize(pd.Series(q_key), sort=False)
    r_codes = pd.Categorical(r_key, categories=uniq).codes
    for code in range(len(uniq)):
        qi = np.flatnonzero(q_codes == code)
        ri = np.flatnonzero(r_codes == code)
        if len(qi):
            yield qi, ri


def count_between(q_key, q_lo, q_hi, r_key, r_t):
    """Events r with q_lo <= r_t < q_hi, same key.  NaN bounds give NaN."""
    out = np.full(len(q_key), np.nan)
    ok_r = ~np.isnan(r_t)
    r_key, r_t = np.asarray(r_key)[ok_r], r_t[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        ts = np.sort(r_t[ri])
        lo, hi = q_lo[qi], q_hi[qi]
        good = ~(np.isnan(lo) | np.isnan(hi))
        res = np.full(len(qi), np.nan)
        res[good] = np.searchsorted(ts, hi[good], "left") - np.searchsorted(ts, lo[good], "left")
        out[qi] = res
    return out


def mean_between(q_key, q_lo, q_hi, r_key, r_t, r_v):
    out = np.full(len(q_key), np.nan)
    ok_r = ~(np.isnan(r_t) | np.isnan(r_v))
    r_key, r_t, r_v = np.asarray(r_key)[ok_r], r_t[ok_r], r_v[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        order = np.argsort(r_t[ri], kind="stable")
        ts, vs = r_t[ri][order], r_v[ri][order]
        cs = np.concatenate([[0.0], np.cumsum(vs)])
        lo, hi = q_lo[qi], q_hi[qi]
        good = ~(np.isnan(lo) | np.isnan(hi))
        a = np.searchsorted(ts, lo[good], "left")
        b = np.searchsorted(ts, hi[good], "left")
        n = b - a
        res = np.full(len(qi), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[good] = np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)
        out[qi] = res
    return out


def mean_between_ex(q_key, q_lo, q_hi, r_key, r_t, r_v, own_v):
    """Sum and count of r values in [q_lo, q_hi) per query (nb_delay_features takes the
    query's own value out when the query is itself one of the r events)."""
    s = np.full(len(q_key), np.nan)
    n = np.zeros(len(q_key))
    ok_r = ~(np.isnan(r_t) | np.isnan(r_v))
    r_key, r_t, r_v = np.asarray(r_key)[ok_r], r_t[ok_r], r_v[ok_r]
    for qi, ri in _by_key(q_key, r_key):
        order = np.argsort(r_t[ri], kind="stable")
        ts, vs = r_t[ri][order], r_v[ri][order]
        cs = np.concatenate([[0.0], np.cumsum(vs)])
        lo, hi = q_lo[qi], q_hi[qi]
        a = np.searchsorted(ts, lo, "left")
        b = np.searchsorted(ts, hi, "left")
        s[qi], n[qi] = cs[b] - cs[a], b - a
    return s, n


def nb_delay_features(F, qk, qt, own, rk, rt, rv, name, windows):
    """Mean of neighbours' values around our takeoff (self excluded), per window spec
    (label, lo offset, hi offset, own-in-window)."""
    for lab, lo, hi, incl in windows:
        s, n = mean_between_ex(qk, qt + lo, qt + hi, rk, rt, rv, own)
        if incl:
            has = ~np.isnan(own)
            s = np.where(has, s - np.nan_to_num(own), s)
            n = np.where(has, n - 1, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            F[f"{name}_{lab}"] = np.where(n >= 2, s / np.maximum(n, 1), np.nan)
        F[f"{name}_n_{lab}"] = n


def stand_neighbours(q_key, q_t, r_key, r_t, r_attrs: dict, lags=(1, 2)):
    """For each query, the r events at the same key: last `lags` at or before q_t, and the next after."""
    n = len(q_key)
    out = {f"t_prev{l}": np.full(n, np.nan) for l in lags}
    out["t_next"] = np.full(n, np.nan)
    for name in r_attrs:
        for l in lags:
            out[f"{name}_prev{l}"] = np.full(n, None, dtype=object)
    ok_r = ~np.isnan(r_t)
    r_key, r_t = np.asarray(r_key)[ok_r], r_t[ok_r]
    attrs = {k: np.asarray(v, dtype=object)[ok_r] for k, v in r_attrs.items()}
    for qi, ri in _by_key(q_key, r_key):
        order = ri[np.argsort(r_t[ri], kind="stable")]
        ts = r_t[order]
        pos = np.searchsorted(ts, q_t[qi], "right")
        for l in lags:
            j = pos - l
            good = j >= 0
            out[f"t_prev{l}"][qi[good]] = ts[j[good]]
            for name, arr in attrs.items():
                out[f"{name}_prev{l}"][qi[good]] = arr[order][j[good]]
        good = pos < len(ts)
        out["t_next"][qi[good]] = ts[pos[good]]
    return out


WX_COLS = ["temp_c", "spread_c", "wind_kt", "gust_kt", "vis_km", "ceil_ft", "precip_mm", "wx_snow",
           "wx_freezing", "wx_rain", "wx_fog", "wx_thunder", "wx_shower", "deice_likely"]


def load_metar() -> pd.DataFrame | None:
    roots = [Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("metar_prc2026.parquet"))
            if hits:
                m = pd.read_parquet(hits[0])
                m["t"] = (pd.to_datetime(m["valid_utc"], utc=True) - EPOCH).dt.total_seconds()
                out = []
                for st, g in m.sort_values("t").groupby("station"):
                    g = g.copy()
                    idx = pd.to_datetime(g["valid_utc"], utc=True)
                    roll = g.set_index(idx)
                    g["snow_3h"] = roll["wx_snow"].rolling("3h").max().to_numpy()
                    g["deice_3h"] = roll["deice_likely"].rolling("3h").max().to_numpy()
                    g["tmin_6h"] = roll["temp_c"].rolling("6h").min().to_numpy()
                    out.append(g)
                print(f"metar: {hits[0]} ({len(m):,} reports)", flush=True)
                return pd.concat(out, ignore_index=True)
    print("metar: not found; weather features skipped", flush=True)
    return None


METAR = None


def weather_features(apt: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    """As-of METAR (latest report at or before t) per airport."""
    cols = WX_COLS + ["snow_3h", "deice_3h", "tmin_6h"]
    out = pd.DataFrame(np.nan, index=np.arange(len(t)), columns=cols)
    if METAR is None:
        return out
    q = pd.DataFrame({"apt": apt, "t": t, "i": np.arange(len(t))}).dropna(subset=["t"]).sort_values("t")
    for st, g in METAR.groupby("station"):
        qq = q[q["apt"] == st]
        if qq.empty:
            continue
        m = pd.merge_asof(qq, g[["t"] + cols].sort_values("t"), on="t", direction="backward",
                          tolerance=3 * 3600.0)
        out.loc[m["i"].to_numpy(), cols] = m[cols].to_numpy()
    return out


ARP = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
ADSB = None
ADSB_RAW = ["t_first", "t_gap_start", "t_stand_end", "t_start", "t_move", "t_taxi", "t_takeoff",
            "stopped_s", "n_stops", "ground_dist_m", "start_to_last_m", "n_points", "n_points_seg",
            "had_stand_dwell", "lat_start", "lon_start",
            "t_move_all", "lat_first", "lon_first", "first_to_last_m", "dist_all_m", "stopped_all_s",
            "max_gap_s", "n_gaps", "dwell_first_s", "air_gap_s"]


def load_adsb() -> pd.DataFrame | None:
    roots = [Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            # Re-extracted files (_w600r richest, then _w600) carry more columns: read them first so the dedupe
            # below keeps their version of a departure both files hold.
            hits = sorted(root.rglob("adsb_departures_*.parquet"),
                          key=lambda h: (0 if "_w600r" in h.name else 1 if "_w600" in h.name else 2, h.name))
            if hits:
                a = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
                a = a.drop_duplicates(subset=["hex", "t_takeoff", "apt"]).reset_index(drop=True)
                for c in ADSB_RAW:
                    if c not in a:
                        a[c] = np.nan
                a["cs"] = a["callsign"].fillna("").astype(str).str.strip().str.upper()
                span = pd.to_datetime(a["t_takeoff"], unit="s", utc=True)
                print(f"adsb: {len(hits)} files, {len(a):,} departures, {span.min()} .. {span.max()}", flush=True)
                return a
    print("adsb: not found; ADS-B features skipped", flush=True)
    return None


def canon(s) -> np.ndarray:
    """Canonical callsign: upper case, alphanumerics only, leading zeros of an ICAO flight number
    dropped (DLH0796 -> DLH796)."""
    s = pd.Series(s).fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return s.str.replace(r"^([A-Z]{3})0+(?=[0-9A-Z])", r"\1", regex=True).to_numpy()


def _is_iata(f: pd.Series) -> pd.Series:
    return f.str.match(r"^[A-Z0-9]{2}\d") & ~f.str.match(r"^[A-Z]{3}\d")


def build_iata_map(flight, cs) -> dict:
    """IATA designator (2 chars) -> ICAO prefix (3 letters), learned from rows that carry an IATA flight
    number and an ICAO callsign; a designator needs >= 5 rows and a >= 50% majority."""
    f = pd.Series(flight).fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    c = pd.Series(cs).fillna("").astype(str).str.upper().str.strip()
    ok = _is_iata(f) & c.str.match(r"^[A-Z]{3}")
    tab = pd.DataFrame({"p": f[ok].str[:2].to_numpy(), "q": c[ok].str[:3].to_numpy()}).value_counts().rename("n").reset_index()
    tot = tab.groupby("p")["n"].transform("sum")
    tab = tab[(tab["n"] >= 5) & (tab["n"] >= 0.5 * tot)].drop_duplicates("p")
    return dict(zip(tab["p"], tab["q"]))


def openflights_map() -> dict:
    """Active airlines' IATA -> ICAO from OpenFlights airlines.dat (ODbL 1.0), unique designators only."""
    import io
    import urllib.request
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
        a = pd.read_csv(io.StringIO(raw), header=None, na_values=["\\N"], keep_default_na=False,
                        names=["id", "name", "alias", "iata", "icao", "callsign", "country", "active"])
        a = a[(a["active"] == "Y") & a["iata"].astype(str).str.match(r"^[A-Z0-9]{2}$")
              & a["icao"].astype(str).str.match(r"^[A-Z]{3}$")]
        a = a[~a["iata"].duplicated(keep=False)]
        return dict(zip(a["iata"], a["icao"]))
    except Exception as exc:  # noqa: BLE001
        print(f"openflights: unavailable ({exc}); data-derived IATA map only", flush=True)
        return {}


def icao_alias(flight) -> np.ndarray:
    f = pd.Series(flight).fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    pre = f.str[:2].map(IATA_MAP)
    return canon(np.where(_is_iata(f) & pre.notna(), pre.fillna("") + f.str[2:], ""))


def airline_key(cs, flight) -> np.ndarray:
    """ICAO airline: the callsign prefix when it is three letters, else the flight number's ICAO prefix
    or its mapped IATA designator; unknown designators keep a '?'-marked IATA prefix."""
    c = pd.Series(cs).fillna("").astype(str).str.upper().str.strip()
    f = pd.Series(flight).fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    a = np.where(c.str.match(r"^[A-Z]{3}"), c.str[:3], "")
    fi = np.where(f.str.match(r"^[A-Z]{3}\d"), f.str[:3], f.str[:2].map(IATA_MAP).fillna("").to_numpy())
    a = np.where(a == "", fi, a)
    return np.where(a == "", "?" + f.str[:2].to_numpy(), a).astype(object)


def _mutual(p: pd.DataFrame, cost: str = "cost") -> pd.DataFrame:
    """Greedy one-to-one assignment: repeatedly keep pairs that are each other's cheapest."""
    p = p.sort_values([cost, "i", "j"], kind="stable")
    keep = []
    while len(p):
        bf = p.drop_duplicates("i")
        be = p.drop_duplicates("j")
        m = bf.merge(be[["i", "j"]], on=["i", "j"])
        if not len(m):
            break
        keep.append(m)
        p = p[~p["i"].isin(m["i"]) & ~p["j"].isin(m["j"])]
    return pd.concat(keep, ignore_index=True) if keep else p.iloc[:0].copy()


MATCH_WIN = 900.0     # callsign matches: |takeoff or first-airborne offset| within this
TONLY_WIN = 60.0      # time-only candidates
TONLY_NA_TOL = 30.0   # a time-only match is credible when the first airborne point lines up this well


def match_events(q_apt, q_t, q_alias: list, q_type, e_apt, e_to, e_na, e_keys: list, e_keys2: list, e_type,
                 tag: str = "") -> pd.DataFrame:
    """One-to-one flight <-> ADS-B event assignment per airport.  Returns i (flight), j (event), conf:
    3 = primary callsign, 2 = another callsign the chain broadcast, 1.5 = time-only with the aircraft type
    agreeing and the first airborne point lined up, 1 = other time-only (not this flight's own evidence)."""
    q_apt = np.asarray(q_apt).astype(str)
    e_apt = np.asarray(e_apt).astype(str)
    q_alias = [np.asarray(a_).astype(str) for a_ in q_alias]
    q_t = np.asarray(q_t, dtype=float)
    e_to = np.asarray(e_to, dtype=float)
    e_na = np.asarray(e_na, dtype=float)
    nq, ne = len(q_t), len(e_to)
    cols = ["i", "j", "conf", "dto", "dna"]
    if nq == 0 or ne == 0:
        return pd.DataFrame(columns=cols)
    # Candidate pairs share airport, a key and a 30-min time bucket (either of the event's anchors,
    # flights probe the neighbouring buckets too): joining on the callsign alone pairs a flight with
    # every day its callsign flew and does not fit in memory over 14 months.
    B = 1800.0
    parts = []
    for ap in np.unique(q_apt):
        qi = np.flatnonzero(q_apt == ap)
        ej = np.flatnonzero(e_apt == ap)
        if not len(ej):
            continue
        Q = pd.concat([pd.DataFrame({"i": qi, "k": np.asarray(a_)[qi]}) for a_ in q_alias])
        Q = Q[(Q["k"].str.len() >= 3).to_numpy() & ~np.isnan(q_t[Q["i"].to_numpy()])].drop_duplicates(["i", "k"])
        qb = np.floor(q_t[Q["i"].to_numpy()] / B)
        Q = pd.concat([Q.assign(b=qb + o) for o in (-1, 0, 1)])
        E = pd.concat([pd.DataFrame({"j": ej, "k": np.asarray(k_)[ej], "lvl": lvl})
                       for lvl, keys in ((0, e_keys), (1, e_keys2)) for k_ in keys])
        E = E[(E["k"].str.len() >= 3).to_numpy()].sort_values("lvl", kind="stable").drop_duplicates(["j", "k"])
        jj_ = E["j"].to_numpy()
        E = pd.concat([E.assign(b=np.floor(e_to[jj_] / B)), E.assign(b=np.floor(e_na[jj_] / B))])
        E = E[~np.isnan(E["b"].to_numpy())].drop_duplicates(["j", "k", "b"])
        mm = Q.merge(E, on=["k", "b"])[["i", "j", "lvl"]]
        parts.append(mm.sort_values("lvl", kind="stable").drop_duplicates(["i", "j"]))
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["i", "j", "lvl"])
    m = m.sort_values("lvl", kind="stable").drop_duplicates(["i", "j"])
    ii, jj = m["i"].to_numpy(dtype=int), m["j"].to_numpy(dtype=int)
    dto = e_to[jj] - q_t[ii]
    dna = e_na[jj] - q_t[ii]
    near = (m["lvl"].to_numpy() == 0) & (np.abs(dto) <= 300)
    o_to = float(np.nanmedian(dto[near])) if near.sum() >= 50 else 0.0
    o_na = float(np.nanmedian(dna[near])) if np.isfinite(dna[near]).sum() >= 50 else np.nan
    c = np.fmin(np.abs(dto - o_to), np.abs(dna - o_na))
    m = m.assign(dto=dto, dna=dna, cost=c + 300.0 * m["lvl"].to_numpy())
    m = m[c <= MATCH_WIN]
    cs = _mutual(m[["i", "j", "cost", "lvl", "dto", "dna"]])
    cs["conf"] = np.where(cs["lvl"].to_numpy() == 0, 3.0, 2.0)
    # time-only: flights and events left over, takeoff within TONLY_WIN, one-to-one
    qi = np.setdiff1d(np.arange(nq), cs["i"].to_numpy())
    ej = np.setdiff1d(np.arange(ne), cs["j"].to_numpy())
    qi = qi[~np.isnan(q_t[qi])]
    ej = ej[~np.isnan(e_to[ej])]
    tq = pd.DataFrame({"i": qi, "apt": q_apt[qi], "b": np.floor(q_t[qi] / TONLY_WIN)})
    te = pd.DataFrame({"j": ej, "apt": e_apt[ej], "b0": np.floor((e_to[ej] - o_to) / TONLY_WIN)})
    te = pd.concat([te.assign(b=te["b0"] + o) for o in (-1, 0, 1)])
    mt = tq.merge(te[["j", "apt", "b"]], on=["apt", "b"])
    mt["dto"] = e_to[mt["j"].to_numpy()] - q_t[mt["i"].to_numpy()]
    mt["dna"] = e_na[mt["j"].to_numpy()] - q_t[mt["i"].to_numpy()]
    mt["cost"] = np.abs(mt["dto"].to_numpy() - o_to)
    mt = mt[mt["cost"] <= TONLY_WIN]
    to = _mutual(mt[["i", "j", "cost", "dto", "dna"]])
    qt_ = pd.Series(np.asarray(q_type)[to["i"].to_numpy()]).astype(str).str.upper().to_numpy()
    et_ = pd.Series(np.asarray(e_type)[to["j"].to_numpy()]).astype(str).str.upper().to_numpy()
    lined = np.abs(to["dna"].to_numpy() - o_na) <= TONLY_NA_TOL if np.isfinite(o_na) else np.zeros(len(to), bool)
    to["conf"] = np.where((qt_ == et_) & lined, 1.5, 1.0)
    out = pd.concat([cs[cols], to[cols]], ignore_index=True)
    vc = out["conf"].value_counts().to_dict()
    print(f"match {tag}: flights {nq:,} events {ne:,} offsets to {o_to:.1f} na {o_na:.1f}  conf counts "
          f"{ {k: vc[k] for k in sorted(vc)} }", flush=True)
    return out


def _event_keys(ev: pd.DataFrame, primary: list, secondary: list):
    k1 = [canon(ev[c]) for c in primary if c in ev]
    k2 = []
    for c in secondary:
        if c in ev:
            parts = ev[c].fillna("").astype(str).str.split("|")
            for p in range(int(min(parts.str.len().max() or 0, 8))):
                k2.append(canon(parts.str[p].fillna("").replace("@@@@@@@@", "")))
    return k1, k2


def blank_adsb(X: pd.DataFrame, apt, airports, cols) -> None:
    """Blank ADS-B features in place at airports without 2025 ground coverage."""
    rows = np.isin(apt, airports)
    X.loc[rows, cols] = np.nan
    X.loc[rows, "adsb_found"] = 0.0


def adsb_features(apt, t, cs1, cs2, actype=None) -> pd.DataFrame:
    n = len(t)
    raw = pd.DataFrame(np.nan, index=np.arange(n), columns=ADSB_RAW + ["cs_match"])
    if ADSB is None:
        return raw
    if FS_OPTS.get("match") == "v7":
        a = ADSB.dropna(subset=["t_takeoff"]).reset_index(drop=True)
        k1, k2 = _event_keys(a, ["callsign"], [])
        na = a["t_air"].to_numpy(dtype=float) if "t_air" in a else np.full(len(a), np.nan)
        mm = match_events(apt, t, [canon(cs1), canon(cs2), icao_alias(cs2)],
                          actype if actype is not None else np.full(n, ""), a["apt"].to_numpy(),
                          a["t_takeoff"].to_numpy(dtype=float), na, k1, k2,
                          a["actype"].to_numpy() if "actype" in a else np.full(len(a), ""), tag="adsb v1")
        own = mm[mm["conf"] >= 1.5]
        i = own["i"].to_numpy(dtype=int)
        raw.loc[i, ADSB_RAW] = a.loc[own["j"].to_numpy(dtype=int), ADSB_RAW].to_numpy()
        raw.loc[i, "cs_match"] = own["conf"].to_numpy()
        conf = np.zeros(n)
        conf[mm["i"].to_numpy(dtype=int)] = mm["conf"].to_numpy()
    q = pd.DataFrame({"i": np.arange(n), "apt": apt, "t": t,
                      "cs1": pd.Series(cs1).fillna("").astype(str).str.strip().str.upper().to_numpy(),
                      "cs2": pd.Series(cs2).fillna("").astype(str).str.strip().str.upper().to_numpy()})
    if FS_OPTS.get("match") != "v7":
        a = ADSB[["apt", "cs"] + ADSB_RAW]
        # Nearest same-callsign takeoff within 5 min, per (airport, callsign). An exact join on
        # the callsign pairs each flight with every day it flew and runs out of memory at 7 months.
        aa = a.dropna(subset=["t_takeoff"]).sort_values("t_takeoff")
        parts = []
        for col in ("cs1", "cs2"):
            qq = q[(q[col].str.len() >= 3) & q["t"].notna()].sort_values("t")
            m = pd.merge_asof(qq, aa.rename(columns={"cs": col}), left_on="t", right_on="t_takeoff",
                              by=["apt", col], direction="nearest", tolerance=300.0)
            parts.append(m[m["t_takeoff"].notna()])
        m = pd.concat(parts, ignore_index=True)
        if len(m):
            m["adt"] = (m["t_takeoff"] - m["t"]).abs()
            m = m.sort_values("adt").drop_duplicates("i")
            raw.loc[m["i"].to_numpy(), ADSB_RAW] = m[ADSB_RAW].to_numpy()
            raw.loc[m["i"].to_numpy(), "cs_match"] = 1.0
        rest = q[~q["i"].isin(m["i"] if len(m) else [])].dropna(subset=["t"])
        for ap, g in rest.groupby("apt"):
            aa = a[a["apt"] == ap].sort_values("t_takeoff")
            if aa.empty:
                continue
            mm = pd.merge_asof(g.sort_values("t"), aa.drop(columns=["apt", "cs"]), left_on="t", right_on="t_takeoff",
                               direction="nearest", tolerance=60.0)
            mm = mm[mm["t_takeoff"].notna()]
            raw.loc[mm["i"].to_numpy(), ADSB_RAW] = mm[ADSB_RAW].to_numpy()
            raw.loc[mm["i"].to_numpy(), "cs_match"] = 0.0
    out = pd.DataFrame(index=raw.index)
    out["adsb_found"] = raw["t_takeoff"].notna().astype(float)
    out["adsb_cs_match"] = raw["cs_match"]
    if FS_OPTS.get("match") == "v7":
        out["adsb_cs_match"] = (raw["cs_match"] >= 2).astype(float).where(raw["cs_match"].notna())
        out["adsb_match_conf"] = conf
        out["adsb_tonly"] = (conf == 1.0).astype(float)
    out["adsb_dt_takeoff"] = raw["t_takeoff"] - t
    for c in ("t_first", "t_gap_start", "t_stand_end", "t_start", "t_move", "t_taxi", "t_move_all"):
        out[f"adsb_obs_{c[2:]}"] = t - raw[c]
    for c in ("stopped_s", "n_stops", "ground_dist_m", "start_to_last_m", "n_points", "n_points_seg", "had_stand_dwell",
              "first_to_last_m", "dist_all_m", "stopped_all_s", "max_gap_s", "n_gaps", "dwell_first_s", "air_gap_s"):
        out[f"adsb_{c}"] = raw[c]
    lat0 = np.array([ARP.get(x, (np.nan, np.nan))[0] for x in apt])
    lon0 = np.array([ARP.get(x, (np.nan, np.nan))[1] for x in apt])
    out["adsb_dy_km"] = (raw["lat_start"] - lat0) * 111.2
    out["adsb_dx_km"] = (raw["lon_start"] - lon0) * 111.2 * np.cos(np.radians(lat0))
    out["adsb_first_dy_km"] = (raw["lat_first"] - lat0) * 111.2
    out["adsb_first_dx_km"] = (raw["lon_first"] - lon0) * 111.2 * np.cos(np.radians(lat0))
    return out


GEOM = None


def load_geometry():
    """Stand / runway / stand x runway tables from research/geometry (OSM ODbL, X-Plane GPL,
    OurAirports public domain), published as usiadianimuwa/prc-taxi-2026-geometry."""
    roots = [Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("stand_runway.parquet"))
            if hits:
                d = hits[0].parent
                g = {"sr": pd.read_parquet(d / "stand_runway.parquet"), "st": pd.read_parquet(d / "stands.parquet"),
                     "rw": pd.read_parquet(d / "runways.parquet")}
                print(f"geometry: {d} ({len(g['sr']):,} stand x runway pairs)", flush=True)
                return g
    print("geometry: not found; skipped", flush=True)
    return None


def geometry_features(apt, stand, rwy) -> pd.DataFrame:
    """Taxi path length, straight distance and runway crossings from the stand to the departure
    runway's end; stand position (km from the ARP), stand heading and ICAO size class; the
    bearing from the stand to the runway end relative to the stand's heading (push direction)."""
    n = len(apt)
    cols = ["geo_path_m", "geo_straight_m", "geo_crossings", "geo_graph_m", "geo_backtrack_m", "geo_path_ratio",
            "geo_stand_dx_km", "geo_stand_dy_km", "geo_stand_hdg_sin", "geo_stand_hdg_cos", "geo_stand_cat",
            "geo_rwy_len_m", "geo_rel_bearing_cos", "geo_rel_bearing_sin", "geo_exact"]
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols)
    if GEOM is None:
        return out
    q = pd.DataFrame({"apt": apt, "stand": pd.Series(stand).astype(str).to_numpy(),
                      "runway": pd.Series(rwy).astype(str).to_numpy()})
    sr = GEOM["sr"][["apt", "stand", "runway", "path_m", "straight_m", "crossings", "graph_m", "backtrack_m",
                     "path_over_straight"]]
    m = q.merge(sr, on=["apt", "stand", "runway"], how="left")
    out["geo_path_m"] = m["path_m"].to_numpy()
    out["geo_straight_m"] = m["straight_m"].to_numpy()
    out["geo_crossings"] = m["crossings"].to_numpy()
    out["geo_graph_m"] = m["graph_m"].to_numpy()
    out["geo_backtrack_m"] = m["backtrack_m"].to_numpy()
    out["geo_path_ratio"] = m["path_over_straight"].to_numpy()
    st = GEOM["st"][["apt", "stand", "lat", "lon", "heading", "xp_icao_cat", "exact"]]
    s = q.merge(st, on=["apt", "stand"], how="left")
    lat0 = np.array([ARP.get(x, (np.nan, np.nan))[0] for x in apt])
    lon0 = np.array([ARP.get(x, (np.nan, np.nan))[1] for x in apt])
    out["geo_stand_dy_km"] = (s["lat"].to_numpy(dtype=float) - lat0) * 111.2
    out["geo_stand_dx_km"] = (s["lon"].to_numpy(dtype=float) - lon0) * 111.2 * np.cos(np.radians(lat0))
    h = np.radians(s["heading"].to_numpy(dtype=float))
    out["geo_stand_hdg_sin"], out["geo_stand_hdg_cos"] = np.sin(h), np.cos(h)
    out["geo_stand_cat"] = s["xp_icao_cat"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}).to_numpy(dtype=float)
    out["geo_exact"] = s["exact"].astype(float).to_numpy()
    rw = GEOM["rw"][["apt", "runway", "thr_lat", "thr_lon", "length_m"]]
    r = q.merge(rw, on=["apt", "runway"], how="left")
    out["geo_rwy_len_m"] = r["length_m"].to_numpy(dtype=float)
    dy = (r["thr_lat"].to_numpy(dtype=float) - s["lat"].to_numpy(dtype=float)) * 111.2
    dx = (r["thr_lon"].to_numpy(dtype=float) - s["lon"].to_numpy(dtype=float)) * 111.2 * np.cos(np.radians(lat0))
    bearing = np.arctan2(dx, dy)
    rel = bearing - h
    out["geo_rel_bearing_cos"], out["geo_rel_bearing_sin"] = np.cos(rel), np.sin(rel)
    return out


def secs(values) -> np.ndarray:
    return (pd.to_datetime(values, utc=True, errors="coerce") - EPOCH).dt.total_seconds().to_numpy()


# %%
def build(df: pd.DataFrame) -> pd.DataFrame:
    """One row per departure, using only fields the ranking file publishes."""
    df = df.reset_index(drop=True)
    dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    arr = df["PHASE_mvt"].eq("ARR").to_numpy()
    apt = np.where(dep, df["ADEP_mvt"].astype(str), df["ADES_mvt"].astype(str))
    rwy = pd.Series(apt) + "|" + df["RUNWAY_mvt"].astype(str)
    t = secs(df["MVT_TIME_UTC_mvt"])
    blk = secs(df["BLOCK_TIME_UTC_mvt"])
    blk[dep] = np.nan  # departure off-block is the answer; blank it in training too
    taxi_in = np.where(arr, blk - t, np.nan)
    aobt3 = secs(df["AOBT_3_flt"])

    d = df.loc[dep].reset_index(drop=True)
    q = np.flatnonzero(dep)
    qa, qt, qr = apt[q], t[q], rwy.to_numpy()[q]
    qa3 = aobt3[q]
    F = pd.DataFrame({ID: d[ID].to_numpy()})
    if TARGET in d and d[TARGET].notna().any():
        F["y"] = d[TARGET].astype("float64").to_numpy()

    # Per-flight timing.
    sched, eobt1, lobt, iobt = (secs(d[c]) for c in ("SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"))
    arvt1, arvt3 = secs(d["ARVT_1_flt"]), secs(d["ARVT_3_flt"])
    F["d_aobt3"] = qt - qa3
    F["d_eobt1"] = qt - eobt1
    F["d_lobt"] = qt - lobt
    F["d_iobt"] = qt - iobt
    F["d_sched"] = qt - sched
    F["dd_utc"] = np.floor(qt / 86400) - np.floor(sched / 86400)
    F["aobt3_m_eobt1"] = qa3 - eobt1
    F["aobt3_m_sched"] = qa3 - sched
    F["lobt_m_iobt"] = lobt - iobt
    F["eobt1_m_lobt"] = eobt1 - lobt
    F["arvt3_m_arvt1"] = arvt3 - arvt1
    F["air3"] = arvt3 - qt
    F["air1"] = arvt1 - eobt1
    F["unmatched"] = np.isnan(qa3).astype("float32")
    F["mvt_sec"] = np.mod(qt, 60)
    F["aobt3_sec"] = np.mod(qa3, 60)
    F["d_aobt3_mod60"] = np.mod(F["d_aobt3"].to_numpy(), 60)

    # Stand occupancy: block-ins at our stand around our takeoff.
    stand_all = (pd.Series(apt) + "|" + df["STAND_mvt"].astype(str)).to_numpy()
    has_stand = df["STAND_mvt"].notna().to_numpy()
    flight_all = df["FLIGHT_mvt"].fillna("").astype(str).str[:3].to_numpy()
    if FS_OPTS.get("airline") == "icao":
        flight_all = airline_key(df["CALLSIGN_flt"], df["FLIGHT_mvt"])
    type_all = df["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    ra = arr & has_stand
    arr_sched_all = secs(df["SCHED_TIME_UTC_mvt"])
    # An arrival whose in-block equals its scheduled time was filled, not observed.
    arr_fill = arr & (np.abs(blk - arr_sched_all) <= 60)
    nb = stand_neighbours(stand_all[q], qt, stand_all[ra], blk[ra],
                          {"air": flight_all[ra], "typ": type_all[ra], "fill": arr_fill[ra].astype(float)})
    my_air, my_typ = flight_all[q], type_all[q]
    q_has_stand = has_stand[q]
    for l in (1, 2):
        gap = qt - nb[f"t_prev{l}"]
        gap[~q_has_stand] = np.nan
        F[f"stand_gap_prev{l}"] = gap
        F[f"stand_same_air_prev{l}"] = np.where(np.isnan(gap), np.nan, (nb[f"air_prev{l}"] == my_air).astype(float))
        F[f"stand_same_typ_prev{l}"] = np.where(np.isnan(gap), np.nan, (nb[f"typ_prev{l}"] == my_typ).astype(float))
    other = (F["stand_same_air_prev1"].to_numpy() == 0) | (F["stand_same_typ_prev1"].to_numpy() == 0)
    F["stand_lb_other"] = np.where(other, F["stand_gap_prev1"].to_numpy(), np.nan)
    nxt = nb["t_next"] - qt
    nxt[~q_has_stand] = np.nan
    F["stand_gap_next"] = nxt
    F["stand_prev1_fill"] = np.array([np.nan if v is None else float(v) for v in nb["fill_prev1"]])
    F["stand_prev1_fill"] = np.where(np.isnan(F["stand_gap_prev1"].to_numpy()), np.nan, F["stand_prev1_fill"].to_numpy())
    # Airport-wide share of filled arrival in-blocks around our scheduled off-block.
    apt_arr = np.where(arr, apt, "")
    for w in (3600, 10800):
        n_all = count_between(qa, sched - w, sched + w, apt_arr[arr], blk[arr])
        n_fill = count_between(qa, sched - w, sched + w, apt_arr[arr_fill], blk[arr_fill])
        F[f"apt_arr_fill_{w // 3600}h"] = np.where(n_all > 0, np.nan_to_num(n_fill) / np.maximum(n_all, 1), np.nan)
    F["stand_arr_2h"] = count_between(stand_all[q], qt - 7200, qt + 1, stand_all[ra], blk[ra])
    F["stand_arr_2h"] = np.where(q_has_stand, F["stand_arr_2h"].to_numpy(), np.nan)

    # Inbound aircraft: latest same-airline block-in at our stand before takeoff.
    stand_air = (pd.Series(stand_all) + "|" + pd.Series(flight_all)).to_numpy()
    ib = stand_neighbours(stand_air[q], qt, stand_air[ra], blk[ra], {"sch": arr_sched_all[ra]}, lags=(1,))
    inb = ib["t_prev1"]
    inb_sch = np.array([np.nan if v is None else float(v) for v in ib["sch_prev1"]])
    inb[~q_has_stand] = np.nan
    dsafe = np.where(F["d_sched"].to_numpy() > 60, F["d_sched"].to_numpy(), np.nan)
    F["inb_gap"] = qt - inb
    F["inb_delay"] = inb - inb_sch
    F["inb_share"] = np.clip((inb - sched) / dsafe, -5, 5)
    F["inb_gap_ratio"] = np.clip((qt - inb) / dsafe, -5, 5)
    F["inb_fill"] = np.where(np.isnan(F["inb_delay"].to_numpy()), np.nan, (np.abs(F["inb_delay"].to_numpy()) <= 60).astype(float))
    if FS_OPTS.get("airline") == "icao":
        # Next occupant: the first in-block at our stand after our (same-airline) inbound's in-block and
        # before our takeoff belongs to another aircraft, so our off-block precedes it (a lower bound on
        # the taxi), unless the stand is shared; how many arrived in between says which.
        inb_d = np.where(qt - inb <= 86400.0, inb, np.nan)  # an inbound older than a day is not ours
        nxo = stand_neighbours(stand_all[q], inb_d + 1.0, stand_all[ra], blk[ra], {}, lags=(1,))["t_next"]
        nxo = np.where(nxo < qt, nxo, np.nan)
        F["stand_next_occ_gap"] = qt - nxo
        F["stand_n_after_inb"] = count_between(stand_all[q], inb_d + 1.0, qt, stand_all[ra], blk[ra])
        F["inb_found"] = (~np.isnan(inb)).astype("float32")

    # Neighbour long-delay context (airport-wide taxiway holds).
    dsch = F["d_sched"].to_numpy()
    longq = dsch > 3600
    safe = np.where(longq, dsch, 1.0)
    c3 = np.where(longq, np.clip(F["d_aobt3"].to_numpy() / safe, 0, 1.5), np.nan)
    ce = np.where(longq, np.clip(F["d_eobt1"].to_numpy() / safe, 0, 1.5), np.nan)
    for w in (3600, 10800):
        F[f"nb_long_share_d3_{w}"] = mean_between(qa, qt - w, qt + w, qa, qt, c3)
        F[f"nb_long_share_e1_{w}"] = mean_between(qa, qt - w, qt + w, qa, qt, ce)
        F[f"nb_long_count_{w}"] = count_between(qa, qt - w, qt + w, qa[longq], qt[longq])

    # Weather at takeoff and around off-block.
    w0 = weather_features(qa, qt)
    w1 = weather_features(qa, qt - 1800)
    for c in w0.columns:
        F[f"wx_{c}_t"] = w0[c].to_numpy()
    for c in ["temp_c", "vis_km", "wind_kt", "deice_likely", "wx_snow", "precip_mm"]:
        F[f"wx_{c}_m30"] = w1[c].to_numpy()

    # ADS-B ground movement (adsb.lol), for the extracted months.
    ad = adsb_features(qa, qt, d["CALLSIGN_flt"].to_numpy(), d["FLIGHT_mvt"].to_numpy(), d["AIRCRAFT_TYPE_mvt"].to_numpy())
    for c in ad.columns:
        F[c] = ad[c].to_numpy()

    # ADS-B v3 events (fs4).
    a3 = adsb3_features(qa, qt, d["CALLSIGN_flt"].to_numpy(), d["FLIGHT_mvt"].to_numpy(), d["STAND_mvt"].to_numpy(),
                        d["AIRCRAFT_TYPE_mvt"].to_numpy())
    for c in a3.columns:
        F[c] = a3[c].to_numpy()

    # Airport geometry (fs3).
    ge = geometry_features(qa, d["STAND_mvt"].to_numpy(), d["RUNWAY_mvt"].to_numpy())
    for c in ge.columns:
        F[c] = ge[c].to_numpy()

    # Local clock.
    local_hour = np.full(len(q), np.nan)
    local_dow = np.full(len(q), np.nan)
    local_day = np.empty(len(q), dtype=object)
    ts_utc = pd.to_datetime(qt, unit="s", utc=True)
    for code, tz in TZ.items():
        m = qa == code
        if m.any():
            lt = ts_utc[m].tz_convert(tz)
            local_hour[m] = lt.hour + lt.minute / 60.0
            local_dow[m] = lt.dayofweek
            local_day[m] = lt.strftime("%Y-%m-%d")
    other = pd.isna(local_day)
    local_day[other] = ts_utc[other].strftime("%Y-%m-%d")
    local_hour[other] = ts_utc[other].hour
    local_dow[other] = ts_utc[other].dayofweek
    F["local_hour"] = local_hour
    F["local_dow"] = local_dow
    F["month"] = ts_utc.month.to_numpy()
    F["mday"] = ts_utc.day.to_numpy()
    day_key = pd.Series(qa) + "|" + pd.Series(local_day)

    # Runway queue and airport flow, from published takeoff and landing times.
    dep_t, dep_k, dep_r = t[dep], apt[dep], rwy.to_numpy()[dep]
    arr_t, arr_k, arr_r = t[arr], apt[arr], rwy.to_numpy()[arr]
    for w in (300, 600, 900, 1800, 3600):
        F[f"dep_prev_{w}"] = count_between(qa, qt - w, qt, dep_k, dep_t)
        F[f"arr_prev_{w}"] = count_between(qa, qt - w, qt, arr_k, arr_t)
    for w in (300, 600, 1800):
        F[f"dep_next_{w}"] = count_between(qa, qt + 1, qt + w, dep_k, dep_t)
        F[f"arr_next_{w}"] = count_between(qa, qt + 1, qt + w, arr_k, arr_t)
    for w in (600, 1800):
        F[f"rwy_dep_prev_{w}"] = count_between(qr, qt - w, qt, dep_r, dep_t)
        F[f"rwy_dep_next_{w}"] = count_between(qr, qt + 1, qt + w, dep_r, dep_t)
        F[f"rwy_arr_prev_{w}"] = count_between(qr, qt - w, qt, arr_r, arr_t)
    lo = np.where(qa3 < qt, qa3, np.nan)
    F["queue_aobt3"] = count_between(qa, lo, qt, dep_k, dep_t)
    F["rwy_queue_aobt3"] = count_between(qr, lo, qt, dep_r, dep_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["queue_rate"] = F["queue_aobt3"] / np.maximum(qt - lo, 60) * 3600

    # Surface load: departures between AOBT_3 and takeoff, arrivals between
    # landing and in-block.
    a3_dep = aobt3[dep]
    ok3 = ~np.isnan(a3_dep)
    far = -1e12
    for name, at in (("at_t", qt), ("at_aobt3", qa3)):
        pushed = count_between(qa, np.full(len(q), far), at + 1, dep_k[ok3], a3_dep[ok3])
        gone = count_between(qa, np.full(len(q), far), at + 1, dep_k[ok3], dep_t[ok3])
        F[f"dep_surface_{name}"] = pushed - gone
        landed = count_between(qa, np.full(len(q), far), at + 1, arr_k, arr_t)
        blocked = count_between(qa, np.full(len(q), far), at + 1, arr_k, blk[arr])
        F[f"arr_surface_{name}"] = landed - blocked

    # Live conditions: arrival taxi-in and neighbours' MVT - AOBT_3.
    tin = np.clip(taxi_in[arr], 0, TE_CLIP)
    for w in (1800, 3600):
        F[f"taxiin_prev_{w}"] = mean_between(qa, qt - w, qt, arr_k, arr_t, tin)
    F["taxiin_pm_1800"] = mean_between(qa, qt - 1800, qt + 1800, arr_k, arr_t, tin)
    d3_all = np.clip(t - aobt3, 0, 5400)
    d3_dep = np.where(dep, d3_all, np.nan)[dep]
    for w in (1800, 3600):
        F[f"nb_d3_pm_{w}"] = mean_between(qa, qt - w, qt + w, dep_k, dep_t, d3_dep)
    F["rwy_nb_d3_pm_1800"] = mean_between(qr, qt - 1800, qt + 1800, dep_r, dep_t, d3_dep)
    # Neighbours' takeoff - EOBT_1 (how late the traffic around us leaves against its plan), self
    # excluded; "x" subtracts each flight's stand x runway median (label-free: both times are
    # published).  At LTFM, with no ADS-B, the same-runway +-15 min mean correlates ~0.2 with the
    # v16 stack's residual.
    e1_dep = np.where(np.abs(qt - eobt1) < 7200, qt - eobt1, np.nan)
    sr = (pd.Series(qr) + "|" + d["STAND_mvt"].astype(str)).to_numpy()
    typ = pd.Series(e1_dep).groupby(sr).transform("median").to_numpy()
    typ = np.where(np.isnan(typ), pd.Series(e1_dep).groupby(qr).transform("median").to_numpy(), typ)
    x1_dep = e1_dep - typ
    wins = [("r600", -600, 600, True), ("r900", -900, 900, True), ("rb900", -900, 0, False)]
    nb_delay_features(F, qr, qt, e1_dep, qr, qt, e1_dep, "nbe1", wins)
    nb_delay_features(F, qr, qt, x1_dep, qr, qt, x1_dep, "nbx1", wins)
    nb_delay_features(F, qa, qt, x1_dep, qa, qt, x1_dep, "nbx1a", [("a900", -900, 900, True)])

    # Airport-day conditions (de-icing, weather, closures).
    arr_day = np.empty(arr.sum(), dtype=object)
    at_utc = pd.to_datetime(arr_t, unit="s", utc=True)
    for code, tz in TZ.items():
        m = arr_k == code
        if m.any():
            arr_day[m] = at_utc[m].tz_convert(tz).strftime("%Y-%m-%d")
    arr_day[pd.isna(arr_day)] = at_utc[pd.isna(arr_day)].strftime("%Y-%m-%d")
    arr_day_key = pd.Series(arr_k) + "|" + pd.Series(arr_day)
    day_tin = pd.Series(tin).groupby(arr_day_key.to_numpy()).mean()
    F["day_taxiin"] = day_key.map(day_tin).to_numpy()
    F["day_d3"] = pd.Series(d3_dep).groupby(day_key.to_numpy()).transform("mean").to_numpy()
    F["day_deps"] = day_key.map(day_key.value_counts()).to_numpy()
    F["day_unmatched"] = pd.Series(F["unmatched"].to_numpy()).groupby(day_key.to_numpy()).transform("mean").to_numpy()

    # Runway configuration per airport-hour (departure and arrival runways in use).
    hb = np.floor(t / 3600)
    ev = pd.DataFrame({"apt": apt, "hb": hb, "rwy": df["RUNWAY_mvt"].astype(str), "dep": dep})
    ev = ev.drop_duplicates()
    conf = (ev.sort_values("rwy").groupby(["apt", "hb", "dep"])["rwy"].agg("/".join).unstack("dep"))
    conf.columns = ["arr_rwys" if not c else "dep_rwys" for c in conf.columns]
    key = pd.MultiIndex.from_arrays([qa, hb[q]])
    conf = conf.reindex(key)
    F["rwy_conf"] = (pd.Series(qa) + "|" + conf["dep_rwys"].fillna("-").to_numpy()
                     + "|" + conf["arr_rwys"].fillna("-").to_numpy()).to_numpy()

    # Categoricals.
    F["apt"] = qa
    F["rwy"] = qr
    F["stand"] = (pd.Series(qa) + "|" + d["STAND_mvt"].astype(str)).to_numpy()
    F["stand_rwy"] = (F["stand"] + "|" + d["RUNWAY_mvt"].astype(str)).to_numpy()
    F["actype"] = d["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    F["wake"] = d["WK_TBL_CAT_flt"].astype(str).to_numpy()
    F["segment"] = d["MARKET_SEGMENT_flt"].astype(str).to_numpy()
    F["ftype"] = d["FLIGHT_TYPE_flt"].astype(str).to_numpy()
    F["frule"] = d["FLIGHT_RULE_mvt"].astype(str).to_numpy()
    F["operator"] = d["AIRCRAFT_OPERATOR_flt"].astype(str).to_numpy()
    F["ades"] = d["ADES_mvt"].astype(str).to_numpy()
    F["ades_region"] = d["ADES_mvt"].astype(str).str[:2].to_numpy()
    F["diverted_plan"] = (d["ADES_FILED_flt"].astype(str) != d["ADES_flt"].astype(str)).astype("float32").to_numpy()
    flight = d["FLIGHT_mvt"].fillna("").astype(str)
    F["flight_len"] = flight.str.len().to_numpy()
    F["flight_dbl"] = ((flight.str.len() >= 5) & (flight.str[1:3] == flight.str[3:5])).astype("float32").to_numpy()
    F["flight_digits"] = flight.str.count(r"\d").to_numpy()
    F["no_actype"] = d["AIRCRAFT_TYPE_mvt"].isna().astype("float32").to_numpy()
    F["no_stand"] = d["STAND_mvt"].isna().astype("float32").to_numpy()
    F["callsign_eq_flight"] = (d["CALLSIGN_flt"].astype(str) == flight).astype("float32").to_numpy()
    F["airline3"] = (pd.Series(qa) + "|" + flight.str[:3]).to_numpy()
    if FS_OPTS.get("airline") == "icao":
        F["airline3"] = (pd.Series(qa) + "|" + pd.Series(flight_all[q])).to_numpy()
    F["flight_key"] = (pd.Series(qa) + "|" + flight).to_numpy()
    F["apt_actype"] = (F["apt"] + "|" + F["actype"]).to_numpy()
    F["apt_operator"] = (F["apt"] + "|" + F["operator"]).to_numpy()
    F["rwy_hour"] = (F["rwy"] + "|" + np.floor(local_hour).astype(int).astype(str)).to_numpy()
    F["t_utc"] = qt
    if FS_OPTS.get("pru"):
        ym = pd.Series(qa) + "|" + pd.Series(ts_utc.year.astype(str)) + "|" + pd.Series(ts_utc.month.astype(str))
        for c in ("pru_pub", "pru_ref", "pru_deice_sh", "pru_noref_sh", "pru_pub_dev"):
            F[c] = ym.map({k: v.get(c, np.nan) for k, v in PRU_TAB.items()}).to_numpy(dtype=float)
    if FS_OPTS.get("demean"):
        # Airport-month-demeaned NM lags: NM clock conventions drift between years (LTFM Jan MVT - EOBT_1
        # +186 s while MVT - AOBT_3 stayed flat; LFPG Jul EOBT_1 changed meaning).  The median is over the
        # frame's own departures of that airport-month with d_sched <= 1 h, so no label is involved.
        apm = pd.Series(qa) + "|" + pd.Series(ts_utc.strftime("%Y-%m"))
        ok = F["d_sched"].to_numpy() <= 3600
        for c in ("d_eobt1", "d_aobt3", "d_lobt", "d_iobt", "aobt3_m_eobt1"):
            v = F[c].to_numpy(dtype=float)
            vv = np.where(ok & (np.abs(v) < 7200), v, np.nan)
            med = pd.Series(vv).groupby(apm.to_numpy()).transform("median").to_numpy()
            F[f"apm_{c}"] = med
            F[f"dm_{c}"] = v - med
    return F


CATS = ["apt", "rwy", "stand", "stand_rwy", "actype", "wake", "segment", "ftype", "frule", "airline3",
        "operator", "ades", "ades_region", "rwy_conf", "apt_actype", "apt_operator"]
TE_KEYS = ["stand_rwy", "stand", "rwy", "apt_actype", "apt_operator", "rwy_hour", "rwy_conf"]
DROP = {ID, "y", "month", "mday", "rwy_hour", "flight_key", "t_utc"}
LONG_KEYS = ["airline3", "flight_key", "apt_operator", "stand_rwy", "apt", "ades"]


# %%
def target_encode(train: dict, month: np.ndarray, y: np.ndarray, fit: np.ndarray,
                  apply: list[dict], m: float = 30.0):
    """Smoothed mean of clipped y over raw (uncollapsed) keys.

    Training rows leave their own month out; `apply` rows use every fit row.
    """
    y = np.clip(y, 0, TE_CLIP)
    yf = np.where(fit, y, np.nan)
    prior_tab = pd.Series(yf).groupby(train["apt"]).mean()
    enc_train = {}
    enc_apply = [dict() for _ in apply]
    for key in TE_KEYS:
        k = train[key]
        frame = pd.DataFrame({"k": k, "mo": month, "y": yf, "n": fit.astype(float)})
        tot = frame.groupby("k")[["y", "n"]].sum()
        km = frame.groupby(["k", "mo"])[["y", "n"]].sum()
        idx = pd.MultiIndex.from_arrays([k, frame["mo"].to_numpy()])
        s = pd.Series(k).map(tot["y"]).to_numpy() - km["y"].reindex(idx).to_numpy()
        n = pd.Series(k).map(tot["n"]).to_numpy() - km["n"].reindex(idx).to_numpy()
        prior = pd.Series(train["apt"]).map(prior_tab).to_numpy()
        enc_train[f"te_{key}"] = (s + m * prior) / (n + m)
        enc_train[f"te_n_{key}"] = n
        for j, keys_apply in enumerate(apply):
            ka = keys_apply[key]
            s_a = pd.Series(ka).map(tot["y"]).fillna(0).to_numpy()
            n_a = pd.Series(ka).map(tot["n"]).fillna(0).to_numpy()
            p_a = pd.Series(keys_apply["apt"]).map(prior_tab).to_numpy()
            enc_apply[j][f"te_{key}"] = (s_a + m * p_a) / (n_a + m)
            enc_apply[j][f"te_n_{key}"] = n_a
    return pd.DataFrame(enc_train), [pd.DataFrame(e) for e in enc_apply]


def categorize(train: pd.DataFrame, others: list[pd.DataFrame], ref_mask: np.ndarray):
    """Rare and unseen levels collapse to one bucket; codes shared across frames."""
    for c in CATS:
        vc = pd.Series(train.loc[ref_mask, c]).value_counts()
        keep = vc.index[vc >= RARE_MIN]
        cats = pd.Index(list(keep) + ["__RARE__"])
        for frame in [train] + others:
            v = frame[c].where(frame[c].isin(keep), "__RARE__")
            frame[c] = pd.Categorical(v, categories=cats)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


QUANTILE_KEYS = ["stand_rwy", "stand"]
QUANTILES = (0.1, 0.5, 0.9)
PATH_DS = 3600.0   # schedule-fill test applies when takeoff is > 1 h past schedule
PATH_TOL = 120.0


def quantile_encode(train: dict, month: np.ndarray, y: np.ndarray, fit: np.ndarray,
                    apply: list[dict], min_n: int = 20):
    """p10/p50/p90 of clipped y per key; training rows leave their own month out."""
    y = np.clip(y, 0, TE_CLIP)
    n = len(y)
    enc_train = {f"tq{int(q * 100)}_{k}": np.full(n, np.nan) for k in QUANTILE_KEYS for q in QUANTILES}
    enc_apply = [{c: np.full(len(a["apt"]), np.nan) for c in enc_train} for a in apply]

    def table(src: np.ndarray, key: str) -> pd.DataFrame:
        frame = pd.DataFrame({"k": train[key][src], "y": y[src]})
        grouped = frame.groupby("k")["y"]
        tab = grouped.quantile(list(QUANTILES)).unstack()
        return tab[grouped.size().reindex(tab.index) >= min_n]

    for key in QUANTILE_KEYS:
        for m in np.unique(month):
            dst = month == m
            tab = table(fit & ~dst, key)
            keys = pd.Series(train[key][dst])
            for q in QUANTILES:
                enc_train[f"tq{int(q * 100)}_{key}"][dst] = keys.map(tab[q]).to_numpy()
        tab = table(fit, key)
        for j, a in enumerate(apply):
            keys = pd.Series(a[key])
            for q in QUANTILES:
                enc_apply[j][f"tq{int(q * 100)}_{key}"] = keys.map(tab[q]).to_numpy()
    return pd.DataFrame(enc_train), [pd.DataFrame(e) for e in enc_apply]



LONG_DS = 3600.0
TWIN_TEST_LIKE = 344.601  # v14 and v18 twin on the days-16-31 hold-out
ADSB_MIN_TRAIN_RATE = 0.0  # v16: ADS-B transfers to airports unseen with it (198.97 -> 193.73), so mask nothing
PATH_TOL = 120.0
R_CLIP = 1.5
W_CAP = 43200.0


def loo_encode(train: dict, month: np.ndarray, values: np.ndarray, fit: np.ndarray, apply: dict,
               keys: list[str], prefix: str, m: float = 20.0):
    """Smoothed mean of `values` per key; training rows leave their own month out."""
    vf = np.where(fit, values, np.nan)
    prior_tab = pd.Series(vf).groupby(train["apt"]).mean()
    prior_all = np.nanmean(vf)
    enc_tr, enc_ap = {}, {}
    prior_tr = pd.Series(train["apt"]).map(prior_tab).fillna(prior_all).to_numpy()
    prior_ap = pd.Series(apply["apt"]).map(prior_tab).fillna(prior_all).to_numpy()
    for key in keys:
        k = train[key]
        frame = pd.DataFrame({"k": k, "mo": month, "v": vf, "n": fit.astype(float)})
        tot = frame.groupby("k")[["v", "n"]].sum()
        km = frame.groupby(["k", "mo"])[["v", "n"]].sum()
        idx = pd.MultiIndex.from_arrays([k, month])
        s = pd.Series(k).map(tot["v"]).to_numpy() - km["v"].reindex(idx).to_numpy()
        n = pd.Series(k).map(tot["n"]).to_numpy() - km["n"].reindex(idx).to_numpy()
        enc_tr[f"{prefix}_{key}"] = (s + m * prior_tr) / (n + m)
        enc_tr[f"{prefix}_n_{key}"] = n
        ka = pd.Series(apply[key])
        s_a = ka.map(tot["v"]).fillna(0).to_numpy()
        n_a = ka.map(tot["n"]).fillna(0).to_numpy()
        enc_ap[f"{prefix}_{key}"] = (s_a + m * prior_ap) / (n_a + m)
        enc_ap[f"{prefix}_n_{key}"] = n_a
    return pd.DataFrame(enc_tr), pd.DataFrame(enc_ap)


def all_encodings(keys_tr, month, y, ds, r, long, path, base_fit, keys_rk):
    te_tr, (te_rk,) = target_encode(keys_tr, month, y, base_fit & ~path, [keys_rk])
    tq_tr, (tq_rk,) = quantile_encode(keys_tr, month, y, base_fit & ~path, [keys_rk])
    lr_tr, lr_rk = loo_encode(keys_tr, month, r, base_fit & long, keys_rk, LONG_KEYS, "ter")
    lp_tr, lp_rk = loo_encode(keys_tr, month, path.astype(float), base_fit & long, keys_rk, LONG_KEYS, "tep")
    return (pd.concat([te_tr, tq_tr, lr_tr, lp_tr], axis=1),
            pd.concat([te_rk, tq_rk, lr_rk, lp_rk], axis=1))



RATIO_SOURCES = ["adsb_obs_move_all", "adsb_stopped_all_s", "adsb_obs_start", "adsb_obs_move", "adsb_obs_first", "adsb_obs_taxi", "adsb_stopped_s", "d_aobt3", "d_eobt1", "d_lobt", "d_iobt", "aobt3_m_sched", "aobt3_m_eobt1",
                 "stand_gap_prev1", "stand_gap_prev2", "stand_lb_other", "nb_d3_pm_1800", "rwy_nb_d3_pm_1800",
                 "te_stand_rwy", "tq10_stand_rwy", "tq50_stand_rwy", "tq90_stand_rwy", "day_taxiin"]


def add_ratios(X: pd.DataFrame) -> pd.DataFrame:
    d = X["d_sched"].to_numpy(dtype=float)
    d = np.where(d > 60, d, np.nan)
    extra = {f"ratio_{c}": np.clip(X[c].to_numpy(dtype=float) / d, -5, 5) for c in RATIO_SOURCES if c in X}
    extra["log_d_sched"] = np.log1p(np.clip(X["d_sched"].to_numpy(dtype=float), 0, None))
    extra = {k: v.astype(np.float32) for k, v in extra.items()}
    return pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1)




# %%
# Label-free congestion measured by ADS-B (fs2).  A departure's excess is its observed taxi
# (takeoff - first movement) minus the typical observed taxi for its stand x runway, where
# "typical" is the median over every credible ADS-B departure of train AND ranking rows (no
# target involved).  The neighbour excess is the mean excess of the other ADS-B departures
# at the same airport (or runway) within +-w of our takeoff.  On 2025 hold-out rows without
# their own ADS-B it tracks y - te_stand_rwy almost 1:1 above +400 s, and on disrupted days
# (EHAM 3-7 Jan 2026 de-icing) it lies far outside 2025's range, where a tree cannot follow
# but an anchor can.
NB_OBS = "adsb_obs_move_all"
NB_MAX = 3600.0


def _credible(F: pd.DataFrame) -> np.ndarray:
    if NB_OBS not in F:  # no ADS-B mounted (smoke slices)
        F[NB_OBS] = np.nan
        F["adsb_found"] = 0.0
    o = F[NB_OBS].to_numpy(dtype=float)
    return (F["adsb_found"].to_numpy() > 0) & (o > 30) & (o < NB_MAX)


def add_nb_excess(frames: list) -> None:
    for F in frames:
        _credible(F)  # creates empty ADS-B columns when none is mounted
    pool = pd.concat([pd.DataFrame({"k": F["stand_rwy"].astype(str).to_numpy(),
                                    "o": F[NB_OBS].to_numpy(dtype=float)})[_credible(F)] for F in frames])
    typ = pool.groupby("k")["o"].median()
    cnt = pool.groupby("k").size()
    typ = typ[cnt >= 10]
    for F in frames:
        ok = _credible(F)
        # period-specific typical taxi (this frame's own months: 2025 for training rows, Jan/Jul
        # 2026 for ranking rows), per stand x runway and per stand x runway x month, leaving the
        # row's own observation out; shows 2026 route changes that 2025 target encodings cannot
        o = np.where(ok, F[NB_OBS].to_numpy(dtype=float), np.nan)
        for name, key in (("sr", F["stand_rwy"].astype(str)),
                          ("srm", F["stand_rwy"].astype(str) + "|" + F["month"].astype(str))):
            fr = pd.DataFrame({"k": key.to_numpy(), "o": o})
            s = fr.groupby("k")["o"].transform("sum").to_numpy() - np.nan_to_num(o)
            n = fr.groupby("k")["o"].transform("count").to_numpy() - ok
            F[f"adsb_per_{name}"] = np.where(n >= 5, s / np.maximum(n, 1), np.nan)
            F[f"adsb_per_n_{name}"] = n
        t = F["t_utc"].to_numpy(dtype=float)
        tp = F["stand_rwy"].astype(str).map(typ).to_numpy(dtype=float)
        exc = np.where(ok, F[NB_OBS].to_numpy(dtype=float) - tp, np.nan)
        F["adsb_typ_sr"] = tp
        F["adsb_exc"] = exc
        for key in ("apt", "rwy"):
            k = F[key].astype(str).to_numpy()
            for w in (1800, 3600):
                s = mean_between(k, t - w, t + w, k, t, exc)
                n = count_between(k, t - w, t + w, k[~np.isnan(exc)], t[~np.isnan(exc)])
                # leave the row's own excess out
                own = np.where(np.isnan(exc), 0.0, exc)
                own_n = (~np.isnan(exc)).astype(float)
                tot = np.nan_to_num(s) * np.nan_to_num(n)
                n2 = np.nan_to_num(n) - own_n
                F[f"nbx_{key}_{w}"] = np.where(n2 > 0, (tot - own) / np.maximum(n2, 1), np.nan)
                F[f"nbn_{key}_{w}"] = n2


# %%
# ADS-B v3 events (solutions/adsb/extract_v3.py + events_v3.py, adsb.lol ODbL): one row per
# departure with graded off-block evidence.  Grade A = seen parked, then leaving (t_obs = the
# pushback, median error -7 s on Jan 2025 labels); B/C = first seen moving (a lower bound) with
# alternative candidates (t_push, t_leave, dwell ends).  Times become "seconds before takeoff".
ADSB3 = None
A3_TIMES = ["t_obs", "t_push", "t_leave", "t_last_dwell_end", "t_longest_dwell_end", "t_dep_first", "t_first",
            "t_park_start", "t_touchdown", "t_first_dwell_end", "dep_evidence_start", "gap_big_t0", "gap_big_t1",
            "gap_last_t0", "gap_last_t1", "t_lineup"]
A3_VALUES = ["turnaround_s", "park_s", "park_to_lineup_m", "park_short", "park_gaps", "gs_push", "gs_leave",
             "leave_gap_s", "last_dwell_s", "last_dwell_to_lineup_m", "longest_dwell_s", "longest_dwell_to_lineup_m",
             "dep_first_to_lineup_m", "gs_dep_first", "first_to_lineup_m", "gs_first", "move_dist_m", "move_time_s",
             "gs_moving_med", "stopped_s", "stopped_near_rwy_s", "stopped_apron_s", "n_stops", "max_gap_after_obs_s",
             "seen_continuous", "coverage", "n_sep_gaps", "n_big_sep_gaps", "max_gap_s", "n_gaps_300", "n_dwells",
             "n_tows", "gap_big_disp_m", "gap_big_to_lineup_m", "gap_last_disp_m", "gap_last_to_lineup_m",
             "first_dwell_s", "tail_gap_s", "takeoff_seen", "chain_landed", "chain_gnd_flag_frac", "chain_mlat_frac"]
A3_POS = ["lat_park", "lon_park", "lat_dep_first", "lon_dep_first", "lat_first", "lon_first"]
A3_ID = ["t_next_air", "cs_next_air", "cs_last", "callsigns", "actype", "obs_taxi_s", "hex", "day", "k"]
DWELLS3 = None
STAND_R_M = 100.0   # a dwell within this of the recorded stand's coordinates is the stand (fs7b)


def load_dwells3() -> pd.DataFrame | None:
    """Per-dwell table of the v3 departures (events_v3.py adsb3_dwells_<TAG>.parquet, fs7b)."""
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("adsb3_dwells_*.parquet"))
            if hits:
                d = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
                d = d.drop_duplicates(["hex", "apt", "day", "k", "d_idx"])
                print(f"dwells3: {len(hits)} files, {len(d):,} dwells", flush=True)
                return d
    print("dwells3: not found; stand-constrained park skipped", flush=True)
    return None


def stand_park_features(i_rows, ev: pd.DataFrame, t, apt, stand, n) -> pd.DataFrame:
    """fs7b: the park chosen with the recorded stand.  Among the matched departure's dwells, the last one
    whose position lies within STAND_R_M of the stand's coordinates is the stand; its end is the
    pushback (grade A when the next point follows within 60 s).  Dwells after it and before takeoff are
    holds (remote, de-icing, runway): their total duration and count.  ev: matched event keys per i_rows."""
    cols = ["a3s_found", "a3s_obs_end", "a3s_obs_start", "a3s_dur", "a3s_dist_m", "a3s_leave_gap", "a3s_gs_leave",
            "a3s_before_gap", "a3s_is_park", "a3s_gradeA", "a3s_hold_s", "a3s_n_hold", "a3s_min_dist_m",
            "a3s_n_dwells"]
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols)
    if DWELLS3 is None or GEOM is None or not len(i_rows):
        return out
    st = GEOM["st"][["apt", "stand", "lat", "lon"]].drop_duplicates(["apt", "stand"])
    q = pd.DataFrame({"i": i_rows, "hex": ev["hex"].to_numpy(), "apt": np.asarray(apt)[i_rows],
                      "day": ev["day"].to_numpy(), "k": ev["k"].to_numpy(),
                      "stand": pd.Series(np.asarray(stand)[i_rows]).astype(str).to_numpy(),
                      "t": np.asarray(t, dtype=float)[i_rows]})
    q = q.merge(st, on=["apt", "stand"], how="left")
    d = q.merge(DWELLS3, on=["hex", "apt", "day", "k"], how="inner")
    if not len(d):
        return out
    kx = np.cos(np.radians(d["lat"].to_numpy(dtype=float)))
    d["dist"] = np.hypot((d["d_lat"].to_numpy(dtype=float) - d["lat"].to_numpy(dtype=float)) * 111200.0,
                         (d["d_lon"].to_numpy(dtype=float) - d["lon"].to_numpy(dtype=float)) * 111200.0 * kx)
    d = d[d["d_t1"] < d["t"]]
    g = d.groupby("i")
    out.loc[g.size().index.to_numpy(), "a3s_n_dwells"] = g.size().to_numpy()
    md = g["dist"].min()
    out.loc[md.index.to_numpy(), "a3s_min_dist_m"] = md.to_numpy()
    near = d[d["dist"] <= STAND_R_M].sort_values(["i", "d_t1"]).drop_duplicates("i", keep="last")
    if not len(near):
        return out
    i = near["i"].to_numpy()
    tt = near["t"].to_numpy()
    out.loc[i, "a3s_found"] = 1.0
    out.loc[i, "a3s_obs_end"] = tt - near["d_t1"].to_numpy()
    out.loc[i, "a3s_obs_start"] = tt - near["d_t0"].to_numpy()
    out.loc[i, "a3s_dur"] = (near["d_t1"] - near["d_t0"]).to_numpy()
    out.loc[i, "a3s_dist_m"] = near["dist"].to_numpy()
    out.loc[i, "a3s_leave_gap"] = near["d_leave_gap_s"].to_numpy(dtype=float)
    out.loc[i, "a3s_gs_leave"] = near["d_gs_leave"].to_numpy(dtype=float)
    out.loc[i, "a3s_before_gap"] = near["d_before_big_gap"].to_numpy(dtype=float)
    out.loc[i, "a3s_is_park"] = near["d_is_park"].to_numpy(dtype=float)
    out.loc[i, "a3s_gradeA"] = ((near["d_leave_gap_s"].to_numpy(dtype=float) <= 60)
                                & (near["d_before_big_gap"].to_numpy() == 0)).astype(float)
    # holds after leaving the stand: later dwells of >= 60 s more than 300 m from line-up
    later = d.merge(near[["i", "d_t1"]].rename(columns={"d_t1": "s_t1"}), on="i")
    later = later[(later["d_t0"] > later["s_t1"]) & (later["d_to_lineup_m"] > 300)]
    h = later.assign(dur=later["d_t1"] - later["d_t0"]).groupby("i")["dur"].agg(["sum", "size"])
    out.loc[i, "a3s_hold_s"] = 0.0
    out.loc[i, "a3s_n_hold"] = 0.0
    out.loc[h.index.to_numpy(), "a3s_hold_s"] = h["sum"].to_numpy()
    out.loc[h.index.to_numpy(), "a3s_n_hold"] = h["size"].to_numpy()
    return out


def load_adsb3() -> pd.DataFrame | None:
    # events are a mounted kernel output even when the other inputs were fetched into DATA_ROOT
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob("adsb3_events_*.parquet")) if root.exists() else []
            if hits:
                cols = ["apt", "callsign", "t_takeoff", "grade"] + A3_TIMES + A3_VALUES + A3_POS + A3_ID
                a = pd.concat([pd.read_parquet(h, columns=[c for c in cols if c in pq_columns(h)]) for h in hits],
                              ignore_index=True)
                for c in cols:
                    if c not in a:
                        a[c] = np.nan
                a = a.dropna(subset=["t_takeoff"]).drop_duplicates(subset=["apt", "t_takeoff", "callsign"])
                a["cs"] = a["callsign"].fillna("").astype(str).str.strip().str.upper()
                print(f"adsb3: {len(hits)} files, {len(a):,} departures", flush=True)
                return a
    print("adsb3: not found; v3 features skipped", flush=True)
    return None


def pq_columns(path) -> list:
    import pyarrow.parquet as _pq
    return _pq.ParquetFile(path).schema_arrow.names


EVP_W = (900, 1800)


def event_pool_features(apt, t, own_j) -> pd.DataFrame:
    """Identity-free surface state from every v3 event at the airport (fs7): mean observed taxi of the
    departures ADS-B saw taking off within +-w of our takeoff (all grades, and grade A only), their count,
    and the share seen at the stand.  The row's own matched event (own_j, -1 = none) is left out."""
    n = len(t)
    out = {}
    a = ADSB3
    ea = a["apt"].astype(str).to_numpy()
    et = a["t_takeoff"].to_numpy(dtype=float)
    ob = np.clip(a["obs_taxi_s"].to_numpy(dtype=float), 0, NB_MAX)
    gA = (a["grade"] == "A").to_numpy().astype(float)
    obA = np.where(gA > 0, ob, np.nan)
    has = own_j >= 0
    jj = np.where(has, own_j, 0)
    own_ob = np.where(has, ob[jj], np.nan)
    own_obA = np.where(has, obA[jj], np.nan)
    own_t = np.where(has, et[jj], np.nan)
    qa = np.asarray(apt).astype(str)
    for w in EVP_W:
        for name, v, ov in (("obs", ob, own_ob), ("obsA", obA, own_obA)):
            s, c = mean_between_ex(qa, t - w, t + w, ea, et, v, ov)
            inside = has & ~np.isnan(ov) & (np.abs(own_t - t) < w)
            s = np.where(inside, s - np.nan_to_num(ov), s)
            c = np.where(inside, c - 1, c)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"evp_{name}_{w}"] = np.where(c >= 1, s / np.maximum(c, 1), np.nan)
            out[f"evp_n{name}_{w}"] = c
    return pd.DataFrame(out)


def adsb3_features(apt, t, cs1, cs2, stand, actype=None) -> pd.DataFrame:
    n = len(t)
    names = (["a3_found", "a3_grade", "a3_cs_match", "a3_dt_takeoff"] + [f"a3_obs_{c[2:]}" for c in A3_TIMES]
             + [f"a3_{c}" for c in A3_VALUES] + ["a3_park_dx_km", "a3_park_dy_km", "a3_first_dx_km", "a3_first_dy_km",
                                                  "a3_park_to_stand_m", "a3_first_to_stand_m"])
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=names)
    if ADSB3 is None:
        return out
    q = pd.DataFrame({"i": np.arange(n), "apt": apt, "t": t,
                      "cs1": pd.Series(cs1).fillna("").astype(str).str.strip().str.upper().to_numpy(),
                      "cs2": pd.Series(cs2).fillna("").astype(str).str.strip().str.upper().to_numpy()})
    cols = ["t_takeoff", "grade"] + A3_TIMES + A3_VALUES + A3_POS
    if FS_OPTS.get("match") == "v7":
        a = ADSB3.reset_index(drop=True)
        k1, k2 = _event_keys(a, ["callsign", "cs_next_air", "cs_last"], ["callsigns"])
        mm = match_events(apt, t, [canon(cs1), canon(cs2), icao_alias(cs2)],
                          actype if actype is not None else np.full(n, ""), a["apt"].to_numpy(),
                          a["t_takeoff"].to_numpy(dtype=float), a["t_next_air"].to_numpy(dtype=float), k1, k2,
                          a["actype"].to_numpy(), tag="adsb v3")
        conf = np.zeros(n)
        conf[mm["i"].to_numpy(dtype=int)] = mm["conf"].to_numpy()
        own_j = np.full(n, -1)
        own = mm[mm["conf"] >= 1.5]
        own_j[own["i"].to_numpy(dtype=int)] = own["j"].to_numpy(dtype=int)
        ev = event_pool_features(apt, np.asarray(t, dtype=float), own_j)
        m = a.loc[own["j"].to_numpy(dtype=int), cols].reset_index(drop=True)
        m["i"] = own["i"].to_numpy(dtype=int)
        m["t"] = np.asarray(t, dtype=float)[m["i"].to_numpy()]
        m["csm"] = (own["conf"].to_numpy() >= 2).astype(float)
        out = _a3_fill(out, m, apt, stand)
        out["a3_match_conf"] = conf
        out["a3_tonly"] = (conf == 1.0).astype(float)
        out["a3_dt_next_air"] = np.nan
        out.loc[own["i"].to_numpy(dtype=int), "a3_dt_next_air"] = own["dna"].to_numpy()
        for c in ev.columns:
            out[c] = ev[c].to_numpy()
        if FS_OPTS.get("park") == "stand":
            keys = a.loc[own["j"].to_numpy(dtype=int), ["hex", "day", "k"]].reset_index(drop=True)
            sp = stand_park_features(own["i"].to_numpy(dtype=int), keys, t, apt, stand, n)
            for c in sp.columns:
                out[c] = sp[c].to_numpy()
        return out
    a = ADSB3[["apt", "cs"] + cols].sort_values("t_takeoff")
    parts = []
    for col in ("cs1", "cs2"):
        qq = q[(q[col].str.len() >= 3) & q["t"].notna()].sort_values("t")
        m = pd.merge_asof(qq, a.rename(columns={"cs": col}), left_on="t", right_on="t_takeoff",
                          by=["apt", col], direction="nearest", tolerance=300.0)
        parts.append(m[m["t_takeoff"].notna()].assign(csm=1.0))
    m = pd.concat(parts, ignore_index=True)
    rest = q[~q["i"].isin(m["i"])].dropna(subset=["t"])
    for ap, g in rest.groupby("apt"):
        aa = a[a["apt"] == ap]
        if aa.empty:
            continue
        mm = pd.merge_asof(g.sort_values("t"), aa.drop(columns=["apt", "cs"]), left_on="t", right_on="t_takeoff",
                           direction="nearest", tolerance=60.0)
        parts.append(mm[mm["t_takeoff"].notna()].assign(csm=0.0))
    m = pd.concat(parts, ignore_index=True)
    if not len(m):
        return out
    m["adt"] = (m["t_takeoff"] - m["t"]).abs()
    m = m.sort_values(["csm", "adt"], ascending=[False, True]).drop_duplicates("i")
    return _a3_fill(out, m, apt, stand)


def _a3_fill(out: pd.DataFrame, m: pd.DataFrame, apt, stand) -> pd.DataFrame:
    """Write matched v3 events (m: i, t, csm + event columns) into the a3_* frame."""
    if not len(m):
        return out
    i = m["i"].to_numpy()
    tq = m["t"].to_numpy()
    out.loc[i, "a3_found"] = 1.0
    out.loc[i, "a3_grade"] = m["grade"].map({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}).to_numpy()
    out.loc[i, "a3_cs_match"] = m["csm"].to_numpy()
    out.loc[i, "a3_dt_takeoff"] = m["t_takeoff"].to_numpy() - tq
    for c in A3_TIMES:
        out.loc[i, f"a3_obs_{c[2:]}"] = tq - m[c].to_numpy(dtype=float)
    for c in A3_VALUES:
        out.loc[i, f"a3_{c}"] = m[c].to_numpy(dtype=float)
    ap_i = np.asarray(apt)[i]
    lat0 = np.array([ARP.get(x, (np.nan, np.nan))[0] for x in ap_i])
    lon0 = np.array([ARP.get(x, (np.nan, np.nan))[1] for x in ap_i])
    kx = 111.2 * np.cos(np.radians(lat0))
    out.loc[i, "a3_park_dy_km"] = (m["lat_park"].to_numpy(dtype=float) - lat0) * 111.2
    out.loc[i, "a3_park_dx_km"] = (m["lon_park"].to_numpy(dtype=float) - lon0) * kx
    out.loc[i, "a3_first_dy_km"] = (m["lat_first"].to_numpy(dtype=float) - lat0) * 111.2
    out.loc[i, "a3_first_dx_km"] = (m["lon_first"].to_numpy(dtype=float) - lon0) * kx
    if GEOM is not None:
        st = GEOM["st"][["apt", "stand", "lat", "lon"]]
        s = pd.DataFrame({"apt": ap_i, "stand": pd.Series(np.asarray(stand)[i]).astype(str).to_numpy()}).merge(
            st, on=["apt", "stand"], how="left")
        for src, name in (("park", "a3_park_to_stand_m"), ("first", "a3_first_to_stand_m")):
            dy = (m[f"lat_{src}"].to_numpy(dtype=float) - s["lat"].to_numpy(dtype=float)) * 111200.0
            dx = (m[f"lon_{src}"].to_numpy(dtype=float) - s["lon"].to_numpy(dtype=float)) * kx * 1000.0
            out.loc[i, name] = np.hypot(dx, dy)
    return out


# ==== remote_inputs.py ====
# %%
# Inputs over URLs, for a kernel on an account that cannot mount the primary's private
# datasets and kernel outputs (miscusi).  build_kernel.py mints the URLs: signed GCS URLs for
# dataset files, tokenised kernel-output URLs; neither needs a credential on the VM.

INPUT_URLS = {'prc-taxi-2026-data/ranking.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/submitting.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-01-01_2025-02-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-02-01_2025-03-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-03-01_2025-04-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-04-01_2025-05-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-05-01_2025-06-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-06-01_2025-07-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-07-01_2025-08-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-08-01_2025-09-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-09-01_2025-10-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-10-01_2025-11-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-11-01_2025-12-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-data/training_2025-12-01_2026-01-01.parquet': '<redacted signed URL>', 'prc-taxi-2026-metar/metar_prc2026.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/README.md': '<redacted signed URL>', 'prc-taxi-2026-geometry/build_geometry.py': '<redacted signed URL>', 'prc-taxi-2026-geometry/check_value.py': '<redacted signed URL>', 'prc-taxi-2026-geometry/coverage.csv': '<redacted signed URL>', 'prc-taxi-2026-geometry/graphs.csv': '<redacted signed URL>', 'prc-taxi-2026-geometry/runways.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/stand_features_raw.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/stand_runway.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/stands.parquet': '<redacted signed URL>', 'prc-taxi-2026-geometry/value_check.csv': '<redacted signed URL>', 'adsb/prc-adsb-extract-202501w/adsb_departures_20250101_20250131_w600.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202502r/adsb_departures_20250201_20250228_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202503r/adsb_departures_20250301_20250331_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202504r/adsb_departures_20250401_20250430_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202505r/adsb_departures_20250501_20250531_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202506r/adsb_departures_20250601_20250630_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202507w/adsb_departures_20250701_20250731_w600.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202508r/adsb_departures_20250801_20250831_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202509r/adsb_departures_20250901_20250930_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202510r/adsb_departures_20251001_20251031_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202511r/adsb_departures_20251101_20251130_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202512r/adsb_departures_20251201_20251231_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202601r/adsb_departures_20260101_20260131_w600r.parquet': '<redacted signed URL>', 'adsb/prc-adsb-extract-202607r/adsb_departures_20260701_20260731_w600r.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202501.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202502.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202503.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202504.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202505.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202506.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202507.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202508.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202509.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202510.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202511.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202512.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202601.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_arrivals_202607.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202501.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202502.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202503.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202504.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202505.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202506.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202507.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202508.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202509.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202510.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202511.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202512.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202601.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_202607.parquet': '<redacted signed URL>', 'k/prc-adsb3-events/adsb3_events_summary.json': '<redacted signed URL>'}


def fetch_inputs(root: str = "/tmp/in") -> None:
    import urllib.request
    base = Path(root)
    for rel, url in INPUT_URLS.items():
        out = base / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=600) as resp, open(out, "wb") as fh:
                    while True:
                        chunk = resp.read(8 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"input retry {attempt} {rel}: {exc}", flush=True)
                time.sleep(10 * (attempt + 1))
    os.environ["DATA_ROOT"] = str(base)
    print(f"inputs: {len(INPUT_URLS)} files under {base}  used_h={used_h():.3f}", flush=True)


# ==== fs_main.py ====
# %%
# Feature store: build every departure's features once and write them with the encodings
# for each fold, so trainer kernels load matrices instead of rebuilding them.
#
# Folds: A = Jan + Jul 2025 days 16-31 (the v14-v27 hold-out), B = Jan + Jul days 1-15.
# Together they give out-of-fold predictions for all of Jan + Jul 2025.  "full" fits the
# encodings on every 2025 row for the ranking refit.  Encodings are leave-one-month-out
# for training rows, so each fold's file covers every training row.
#
# Outputs (float32 features, raw strings for categoricals):
#   tr_base.parquet  rk_base.parquet  enc_A.parquet  enc_B.parquet  enc_full.parquet
#   enc_full_rk.parquet  fs_meta.json


def fold_masks(month: np.ndarray, mday: np.ndarray) -> dict:
    jj = np.isin(month, VAL_MONTHS)
    # F: a forward winter fold (Nov + Dec 2025), fitted on Jan-Oct only (see fs_main)
    return {"A": jj & (mday >= 16), "B": jj & (mday <= 15), "F": np.isin(month, (11, 12))}


def _f32(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == np.float64 and c not in ("y", ID):  # ids exceed float32's exact range
            out[c] = out[c].astype(np.float32)
    return out


def fs_main():
    global METAR, ADSB
    if globals().get("INPUT_URLS"):
        fetch_inputs()
    data_dir = resolve_data_dir()
    print(f"data: {data_dir}", flush=True)
    METAR = load_metar()
    ADSB = load_adsb()
    globals()["METAR"], globals()["ADSB"] = METAR, ADSB
    globals()["GEOM"] = load_geometry()
    globals()["ADSB3"] = load_adsb3()
    if FS_OPTS.get("park") == "stand":
        globals()["DWELLS3"] = load_dwells3()
    raw = pd.concat((pd.read_parquet(p) for p in sorted(data_dir.glob("training_*.parquet"))), ignore_index=True)
    ranking = pd.read_parquet(data_dir / "ranking.parquet")
    if FS_OPTS:
        print(f"fs7 options: {FS_OPTS}", flush=True)
        mp = build_iata_map(pd.concat([raw["FLIGHT_mvt"], ranking["FLIGHT_mvt"]]),
                            pd.concat([raw["CALLSIGN_flt"], ranking["CALLSIGN_flt"]]))
        if FS_OPTS.get("openflights"):
            of = openflights_map()
            added = {k: v for k, v in of.items() if k not in mp}
            print(f"iata map: {len(mp)} from data, +{len(added)} from OpenFlights", flush=True)
            mp.update(added)
        IATA_MAP.clear()
        IATA_MAP.update(mp)
        print(f"iata map: {len(IATA_MAP)} designators", flush=True)
    tr = build(raw)
    del raw
    rk = build(ranking)
    if FS_OPTS.get("ext"):
        tr, rk = merge_external(tr, rk, FS_OPTS["ext"])
    if FS_OPTS:
        for name, F in (("tr", tr), ("rk", rk)):
            cols = [c for c in ("a3_match_conf", "adsb_match_conf") if c in F]
            for c in cols:
                print(f"{name} {c}: " + F.groupby("apt")[c].value_counts().unstack(fill_value=0).to_string(), flush=True)
            if "inb_found" in F:
                print(f"{name} inbound found by airport: {F.groupby('apt')['inb_found'].mean().round(3).to_dict()}", flush=True)
    tr = tr[tr["y"].notna()].reset_index(drop=True)
    print(f"built: train deps {len(tr):,}  ranking deps {len(rk):,}  used_h={used_h():.3f}", flush=True)
    add_nb_excess([tr, rk])
    print(f"nb excess: rk nbx_apt_1800 p50/p99 {np.nanpercentile(rk['nbx_apt_1800'], [50, 99])}  used_h={used_h():.3f}", flush=True)

    month = tr["month"].to_numpy()
    y = tr["y"].to_numpy()
    ds = tr["d_sched"].to_numpy()
    long = ds > LONG_DS
    path = long & (np.abs(y - ds) <= PATH_TOL)
    r = np.where(long, np.clip(y / np.where(long, ds, 1.0), 0, R_CLIP), np.nan)
    dd = tr["dd_utc"].to_numpy()
    artifact = (dd >= 1) & ((y - 86400 * dd) >= -900) & ((y - 86400 * dd) <= 3600)
    tr["artifact"] = artifact
    key_names = sorted(set(TE_KEYS + LONG_KEYS + ["apt"]))
    keys_tr = {k: tr[k].astype(str).to_numpy() for k in key_names}
    keys_rk = {k: rk[k].astype(str).to_numpy() for k in key_names}

    _f32(tr).to_parquet(OUTPUT_DIR / "tr_base.parquet", index=False)
    _f32(rk).to_parquet(OUTPUT_DIR / "rk_base.parquet", index=False)
    folds = fold_masks(month, tr["mday"].to_numpy())
    for name, val in folds.items():
        fit = ~val & ~artifact
        if name == "F":
            fit &= month <= 10  # forward: nothing after the validated months
        enc_tr, _ = all_encodings(keys_tr, month, y, ds, r, long, path, fit, keys_rk)
        _f32(enc_tr).to_parquet(OUTPUT_DIR / f"enc_{name}.parquet", index=False)
        print(f"enc_{name}: {enc_tr.shape}  val rows {int(val.sum()):,}  used_h={used_h():.3f}", flush=True)
    enc_all, enc_rk = all_encodings(keys_tr, month, y, ds, r, long, path, ~artifact, keys_rk)
    _f32(enc_all).to_parquet(OUTPUT_DIR / "enc_full.parquet", index=False)
    _f32(enc_rk).to_parquet(OUTPUT_DIR / "enc_full_rk.parquet", index=False)
    meta = {"fs_opts": FS_OPTS, "iata_map_n": len(IATA_MAP), "n_tr": len(tr), "n_rk": len(rk), "folds": {k: int(v.sum()) for k, v in folds.items()},
            "artifact": int(artifact.sum()), "columns": list(tr.columns), "enc_columns": list(enc_all.columns),
            "used_h": used_h()}
    (OUTPUT_DIR / "fs_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"FS_DONE used_h={used_h():.3f}", flush=True)


def merge_external(tr: pd.DataFrame, rk: pd.DataFrame, pattern: str):
    """Left-join an external per-movement table (numeric columns, keyed by MVT_ID_mvt) onto both frames."""
    roots = ([Path(os.environ["DATA_ROOT"])] if os.environ.get("DATA_ROOT") else []) + [Path("/kaggle/input"), Path.cwd()]
    hits = [h for r in roots if r.exists() for h in sorted(r.rglob(pattern))]
    if not hits:
        raise FileNotFoundError(f"ext: no file matches {pattern}")
    ext = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True).drop_duplicates()
    ext[ID] = ext[ID].astype("int64")
    assert ext[ID].is_unique, "ext: MVT_ID_mvt must be unique"
    keep = [c for c in ext.columns if c != ID and pd.api.types.is_numeric_dtype(ext[c])]
    ext = ext[[ID] + keep]
    out = []
    for F in (tr, rk):
        ids = F[ID].astype("int64")
        G = pd.DataFrame({ID: ids}).merge(ext, on=ID, how="left", validate="one_to_one")
        for c in keep:
            F[c] = G[c].to_numpy(dtype=float)
        out.append(F)
        print(f"ext {pattern}: {len(keep)} columns, coverage {G[keep[0]].notna().mean():.3f}" if keep else "ext: none", flush=True)
    return out


if __name__ == "__main__":
    fs_main()
