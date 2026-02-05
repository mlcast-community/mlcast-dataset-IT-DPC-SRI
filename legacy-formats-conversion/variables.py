from simcradarlib.io_utils.structure_class import StructVariable
import numpy as np
from dataclasses import dataclass
from typing import Optional


"""
Sostituisce la tabella /autofs/radar/radar/file_x_idl/tabelle/variabili.txt
e contiene classi figlie di StructVariable, implementata in
simcradarlib.io_utils.structure_class che implementano le variabili radar.
(Per ora non sono implementate Z_50, Z_70 )
"""


@dataclass
class Variable:
    """
    Dataclass per la gestione delle informazioni sulla variabile del BUFR.

    -name                   --str    (default=None)  : nome della variabile.
    -long_name              --str    (default=None)  : nome esteso della variabile.
    -standard_name          --str    (default=None)  : nome standard della variabile, netCDF compliant.
    -units                  --str    (default=None)  : unità fisiche della variabile letta.
    -min_val                --float  (default=None)  : valore minimo variabile.
    -max_val                --float  (default=None)  : valore massimo variabile.
    -missing                --float  (default=None)  : valore degli out of range/dato mancante.
    -undetect               --float  (default=None)  : valore sottosoglia, assegnato ai punti in cui il valore
                                                       rilevato<min_val.
    -color_table            --str    (default=None)  : filename del file txt con livelli e colori per la grafica.
    -accum_time_h           --float  (default=None)  : tempo di cumulata (0 se campo istantaneo).

    ** Nelle strutture per le informazioni sulla variabile ottenute dalle routine IDL erano presenti anche i
       seguenti attributi riportati qui sotto per completezza:
    -offset                 --float  (default=0.)    : valore offset (se il campo va compresso,residuo da idl)
    -scale_factor           --float  (default=0.)    : valore scale_factor (se il campo va compresso,residuo da idl)
    -nbyte
    -val_compresso
    -tab_id
    """

    name: Optional[str] = None
    long_name: Optional[str] = None
    standard_name: Optional[str] = None
    units: Optional[str] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    missing: Optional[np.float32] = None
    undetect: Optional[np.float32] = None
    color_table: Optional[str] = None
    accum_time_h: Optional[float] = None
    offset: Optional[float] = 0.0
    scale_factor: Optional[float] = 0.0
    nbyte: Optional[int] = None
    val_compresso: Optional[np.ndarray] = None
    tab_id: Optional[int] = None


@dataclass
class VarPr(Variable):
    def __post_init__(self):
        self.name = "pr"
        self.long_name = "Radar precipitation flux"
        self.standard_name = "precipitation_flux"
        self.units = "kg m-2 s-1"
        self.min_val = 0.0
        self.max_val = 10000.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(0.0)


@dataclass
class VarZ60(Variable):
    def __post_init__(self):
        self.name = "Z_60"
        self.long_name = "Radar reflectivity factor"
        self.units = "dBZ"
        self.min_val = -19.69
        self.max_val = 60.0
        self.missing = np.float32(-20.0)
        self.undetect = np.float32(-19.69)
        self.color_table = "RGB_Z.txt"


@dataclass
class VarCumPrr(Variable):
    def __post_init__(self):
        self.name = "cum_pr"
        self.long_name = "Radar Precipitation amount"
        self.standard_name = "precipitation_amount"
        self.units = "kg m-2"
        self.min_val = 0.0
        self.max_val = 10000.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(0.0)


@dataclass
class VarZdr(Variable):
    def __post_init__(self):
        self.name = "ZDR"
        self.long_name = "Radar Differential Reflectivity"
        self.units = "dB"
        self.min_val = -6.0
        self.max_val = 10.0
        self.missing = np.float32(-16.0)
        self.undetect = np.float32(-16.0)
        self.color_table = "RGB_ZDR.txt"


@dataclass
class VarVn16(Variable):
    def __post_init__(self):
        self.name = "VN16"
        self.long_name = "Doppler Radar Velocity"
        self.units = "m s-1"
        self.min_val = -16.5
        self.max_val = 16.5
        self.missing = np.float32(-16.5)
        self.undetect = np.float32(-16.5)
        self.color_table = "RGB_V.txt"


@dataclass
class VarVn49(Variable):
    def __post_init__(self):
        self.name = "VN49"
        self.long_name = "Doppler Radar Velocity"
        self.units = "m s-1"
        self.min_val = -49.5
        self.max_val = 49.5
        self.missing = np.float32(-49.5)
        self.undetect = np.float32(-49.5)
        self.color_table = "RGB_V.txt"


@dataclass
class VarSv(Variable):
    def __post_init__(self):
        self.name = "sV"
        self.long_name = "Doppler Radar Velocity Spectrum Width"
        self.units = "m s-1"
        self.min_val = 0.0
        self.max_val = 10.0
        self.missing = np.float32(0.0)
        self.undetect = np.float32(0.0)
        self.color_table = "RGB_SV.txt"


@dataclass
class VarQc(Variable):
    def __post_init__(self):
        self.name = "qc"
        self.long_name = "Quality"
        self.standard_name = "quality"
        self.units = "percent"
        self.min_val = 0.0
        self.max_val = 1.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)


@dataclass
class VarPrmm(Variable):
    def __post_init__(self):
        self.name = "pr_mm"
        self.long_name = "Radar precipitation flux"
        self.standard_name = "precipitation_flux"
        self.units = "mm h-1"
        self.min_val = 0.0
        self.max_val = 10000.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(0.0)


@dataclass
class VarCumPrmm(Variable):
    def __post_init__(self):
        self.name = "cum_pr_mm"
        self.long_name = "Radar precipitation amount"
        self.standard_name = "precipitation_amount"
        self.units = "mm"
        self.min_val = 0.0
        self.max_val = 10000.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(0.0)


@dataclass
class VarZ(Variable):
    def __post_init__(self):
        self.name = "Z"
        self.long_name = "Radar reflectivity factor"
        self.units = "dBZ"
        self.min_val = -64.0
        self.max_val = 80.0
        self.missing = np.float32(-70.0)
        self.undetect = np.float32(-64.0)
        self.color_table = "RGB_Z.txt"


@dataclass
class VarTh(Variable):
    def __post_init__(self):
        self.name = "Z"
        self.long_name = "Uncorrected Radar reflectivity factor"
        self.units = "dBZ"
        self.min_val = -64.0
        self.max_val = 80.0
        self.missing = np.float32(-70.0)
        self.undetect = np.float32(-64.0)
        self.color_table = "RGB_Z.txt"


@dataclass
class VarDbzh(Variable):
    def __post_init__(self):
        self.name = "DBZH"
        self.long_name = "Radar reflectivity factor"
        self.units = "dBZ"
        self.min_val = -64.0
        self.max_val = 80.0
        self.missing = np.float32(-70.0)
        self.undetect = np.float32(-64.0)
        self.color_table = "RGB_Z.txt"


@dataclass
class VarVrad(Variable):
    def __post_init__(self):
        self.name = "VRAD"
        self.long_name = "Doppler Radar Velocity"
        self.units = "m s-1"
        self.min_val = -49.5
        self.max_val = 49.5
        self.missing = np.float32(-49.5)
        self.undetect = np.float32(-49.5)
        self.color_table = "RGB_V_48_17livelli.txt"


@dataclass
class VarWrad(Variable):
    def __post_init__(self):
        self.name = "WRAD"
        self.long_name = "Doppler Radar Velocity Spectrum Width"
        self.units = "m s-1"
        self.min_val = 0.0
        self.max_val = 10.0
        self.missing = np.float32(0.0)
        self.undetect = np.float32(0.0)
        self.color_table = "RGB_SV.txt"


@dataclass
class VarRhohv(Variable):
    def __post_init__(self):
        self.name = "RHOHV"
        self.long_name = "Correlation ZH-ZV"
        self.units = "percent"
        self.min_val = 0.0
        self.max_val = 1.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)
        self.color_table = "RGB_RHO.txt"


@dataclass
class VarPhidp(Variable):
    def __post_init__(self):
        self.name = "PHIDP"
        self.long_name = "Differential phase"
        self.units = "degree"
        self.min_val = -180.0
        self.max_val = 180.0
        self.missing = np.float32(-180.0)
        self.undetect = np.float32(-180.0)


@dataclass
class VarHght(Variable):
    def __post_init__(self):
        self.name = "HGHT"
        self.long_name = "Height"
        self.units = "km"
        self.min_val = -6.0
        self.max_val = 20.0
        self.missing = np.float32(-9999.0)
        self.undetect = np.float32(-9999.0)
        self.color_table = "RGB_GENERAL.txt"


@dataclass
class VarDbzV(Variable):
    def __post_init__(self):
        self.name = "DBZV"
        self.long_name = "Radar reflectivity factor"
        self.units = "dBZ"
        self.min_val = -64.0
        self.max_val = 80.0
        self.missing = np.float32(-70.0)
        self.undetect = np.float32(-64.0)
        self.color_table = "RGB_Z.txt"


@dataclass
class VarPoh(Variable):
    def __post_init__(self):
        self.name = "POH"
        self.long_name = "Probability of Hail"
        self.units = "percent"
        self.min_val = 0.0
        self.max_val = 1.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)
        self.color_table = "RGB_GENERAL.txt"


@dataclass
class VarVil(Variable):
    def __post_init__(self):
        self.name = "VIL"
        self.long_name = "Vertical integrated liquid Water"
        self.units = "km m-2"
        self.min_val = 0.0
        self.max_val = 150.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)
        self.color_table = "RGB_VIL.txt"


@dataclass
class VarClassConv(Variable):
    def __post_init__(self):
        self.name = "CLASS_CONV"
        self.long_name = "Convective-Stratiform class"
        self.units = ""
        self.min_val = 0.0
        self.max_val = 1500.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)
        self.color_table = "RGB_GENERAL.txt"


@dataclass
class VarSnr(Variable):
    def __post_init__(self):
        self.name = "SNR"
        self.long_name = "Signal Noise Ratio"
        self.units = "dB"
        self.min_val = -8.0
        self.max_val = 8.0
        self.missing = np.float32(-8.0)
        self.undetect = np.float32(8.0)
        self.color_table = "RGB_GENERAL.txt"


@dataclass
class VarClass(Variable):
    def __post_init__(self):
        self.name = "CLASS"
        self.long_name = "Hydrometeor Classification"
        self.units = ""
        self.min_val = 0.0
        self.max_val = 11.0
        self.missing = np.float32(12.0)
        self.undetect = np.float32(9.0)
        self.color_table = "RGB_HYDROCLASS.2.txt"


@dataclass
class VarVilDensity(Variable):
    def __post_init__(self):
        self.name = "VILdensity"
        self.long_name = "Hail size"
        self.units = "cm"
        self.min_val = 0.0
        self.max_val = 10.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)
        self.color_table = "RGB_VILdensity.txt"


@dataclass
class VarRate(Variable):
    def __post_init__(self):
        self.name = "RATE"
        self.long_name = "Rain Rate"
        self.standard_name = "precipitation_flux"
        self.units = "mm h-1"
        self.min_val = 0.0
        self.max_val = 10000.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(0.0)
        self.color_table = "RGB_SRI.txt"


@dataclass
class VarAcrr(Variable):
    def __post_init__(self):
        self.name = "ACRR"
        self.long_name = "Accumulated precipitation"
        self.standard_name = "precipitation_amount"
        self.units = "mm"
        self.min_val = 0.0
        self.max_val = 10000.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(0.0)
        self.color_table = "RGB_CUMULATE.txt"


@dataclass
class VarClassId(Variable):
    def __post_init__(self):
        self.name = "ClassID"
        self.long_name = "Fuzzy logic class"
        self.units = ""
        self.min_val = 0.0
        self.max_val = 4.0
        self.missing = np.float32(-1.0)
        self.undetect = np.float32(-1.0)


VARS = {
    "pr": VarPr,
    "Z_60": VarZ60,
    "cum_pr": VarCumPrr,
    "ZDR": VarZdr,
    "VN16": VarVn16,
    "VN49": VarVn49,
    "sV": VarSv,
    "qc": VarQc,
    "pr_mm": VarPrmm,
    "cum_pr_mm": VarCumPrmm,
    "Z": VarZ,
    "Th": VarTh,
    "DBZH": VarDbzh,
    "VRAD": VarVrad,
    "WRAD": VarWrad,
    "RHOHV": VarRhohv,
    "PHIDP": VarPhidp,
    "HGHT": VarHght,
    "DBZV": VarDbzV,
    "POH": VarPoh,
    "VIL": VarVil,
    "CLASS_CONV": VarClassConv,
    "SNR": VarSnr,
    "CLASS": VarClass,
    "VILdensity": VarVilDensity,
    "RATE": VarRate,
    "ACRR": VarAcrr,
    "ClassID": VarClassId,
}
