import subprocess
import os
import tempfile
import tarfile
from pathlib import Path
import magic
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass, field

from variables import Variable, VarZ, VarPrmm, VarCumPrmm
import pyproj
import numpy as np
import io
import sys
import copy


# from pysteps.visualization import plot_precip_field   
def get_colormap(ptype, units="mm/h", colorscale="pysteps"):
    from matplotlib import cm, colors
    import matplotlib.pyplot as plt

    """
    Function to generate a colormap (cmap) and norm.

    Parameters
    ----------
    ptype : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If ptype is 'prob', this specifies the unit of
        the intensity threshold.
    colorscale : {'pysteps', 'STEPS-BE', 'STEPS-NL', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.

    Returns
    -------
    cmap : Colormap instance
        colormap
    norm : colors.Normalize object
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevs_str: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).
    """
    if ptype in ["intensity", "depth"]:
        # Get list of colors
        color_list, clevs, clevs_str = _get_colorlist(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list(
            "cmap", color_list, len(clevs) - 1
        )

        if colorscale == "BOM-RF3":
            cmap.set_over("black", 1)
        if colorscale == "pysteps":
            cmap.set_over("darkred", 1)
        if colorscale == "STEPS-NL":
            cmap.set_over("darkmagenta", 1)
        if colorscale == "STEPS-BE":
            cmap.set_over("black", 1)
        norm = colors.BoundaryNorm(clevs, cmap.N)

        cmap.set_bad("gray", alpha=0.5)
        cmap.set_under("none")

        return cmap, norm, clevs, clevs_str

    if ptype == "prob":
        cmap = copy.copy(plt.get_cmap("OrRd", 10))
        cmap.set_bad("gray", alpha=0.5)
        cmap.set_under("none")
        clevs = np.linspace(0, 1, 11)
        clevs[0] = 1e-3  # to set zeros to transparent
        norm = colors.BoundaryNorm(clevs, cmap.N)
        clevs_str = [f"{clev:.1f}" for clev in clevs]
        return cmap, norm, clevs, clevs_str

    return cm.get_cmap("jet"), colors.Normalize(), None, None


# from pysteps.visualization import plot_precip_field   
def _get_colorlist(units="mm/h", colorscale="pysteps"):
    """
    Function to get a list of colors to generate the colormap.

    Parameters
    ----------
    units : str
        Units of the input array (mm/h, mm or dBZ)
    colorscale : str
        Which colorscale to use (BOM-RF3, pysteps, STEPS-BE, STEPS-NL)

    Returns
    -------
    color_list : list(str)
        List of color strings.

    clevs : list(float)
        List of precipitation values defining the color limits.

    clevs_str : list(str)
        List of precipitation values defining the color limits
        (with correct number of decimals).
    """

    if colorscale == "BOM-RF3":
        color_list = np.array(
            [
                (255, 255, 255),  # 0.0
                (245, 245, 255),  # 0.2
                (180, 180, 255),  # 0.5
                (120, 120, 255),  # 1.5
                (20, 20, 255),  # 2.5
                (0, 216, 195),  # 4.0
                (0, 150, 144),  # 6.0
                (0, 102, 102),  # 10
                (255, 255, 0),  # 15
                (255, 200, 0),  # 20
                (255, 150, 0),  # 30
                (255, 100, 0),  # 40
                (255, 0, 0),  # 50
                (200, 0, 0),  # 60
                (120, 0, 0),  # 75
                (40, 0, 0),
            ]
        )  # > 100
        color_list = color_list / 255.0
        if units == "mm/h":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                6,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                75,
                100,
                150,
            ]
        elif units == "mm":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                5,
                7,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
            ]
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "pysteps":
        # pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
        color_list = [
            redgrey_hex,
            "#640064",
            "#AF00AF",
            "#DC00DC",
            "#3232C8",
            "#0064FF",
            "#009696",
            "#00C832",
            "#64FF00",
            "#96FF00",
            "#C8FF00",
            "#FFFF00",
            "#FFC800",
            "#FFA000",
            "#FF7D00",
            "#E11900",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [
                0.08,
                0.16,
                0.25,
                0.40,
                0.63,
                1,
                1.6,
                2.5,
                4,
                6.3,
                10,
                16,
                25,
                40,
                63,
                100,
                160,
            ]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "STEPS-NL":
        redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
        color_list = [
            "lightgrey",
            "lightskyblue",
            "deepskyblue",
            "blue",
            "darkblue",
            "yellow",
            "gold",
            "darkorange",
            "red",
            "darkred",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [0.1, 0.5, 1.0, 1.6, 2.5, 4.0, 6.4, 10.0, 16.0, 25.0, 40.0]
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "STEPS-BE":
        color_list = [
            "cyan",
            "deepskyblue",
            "dodgerblue",
            "blue",
            "chartreuse",
            "limegreen",
            "green",
            "darkgreen",
            "yellow",
            "gold",
            "orange",
            "red",
            "magenta",
            "darkmagenta",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)

    else:
        print("Invalid colorscale", colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevs_str = _dynamic_formatting_floats(clevs)

    return color_list, clevs, clevs_str


# from pysteps.visualization import plot_precip_field   
def _dynamic_formatting_floats(float_array, colorscale="pysteps"):
    """Function to format the floats defining the class limits of the colorbar."""
    float_array = np.array(float_array, dtype=float)

    labels = []
    for label in float_array:
        if 0.1 <= label < 1:
            if colorscale == "pysteps":
                formatting = ",.2f"
            else:
                formatting = ",.1f"
        elif 0.01 <= label < 0.1:
            formatting = ",.2f"
        elif 0.001 <= label < 0.01:
            formatting = ",.3f"
        elif 0.0001 <= label < 0.001:
            formatting = ",.4f"
        elif label >= 1 and label.is_integer():
            formatting = "i"
        else:
            formatting = ",.1f"

        if formatting != "i":
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels


@dataclass
class Time:
    """
    Dataclass per la gestione delle informazioni temporali del BUFR.

    -date_time_validita   --datetime  (default=None)  : oggetto datetime.datetime che rappresenta
                                                        la dataora di validità dei dati.
    -date_time_emissione  --datetime  (default=None)  : oggetto datetime.datetime che rappresenta
                                                        la dataora di emissione dei dati.
    -acc_time             --int       (default=None)  : tempo di cumulata se il campo non è istantaneo
                                                        ma cumulato. Nell'implementazione attuale il
                                                        tempo di cumulata letto da un file netCDF non è
                                                        assegnato a questo attributo, ma all'attributo
                                                        accum_time_h di StructVariable, attenendosi alle
                                                        precedenti routine IDL.
    -acc_time_unit        --str       (default=None)  : unità del tempo di cumulazione. Es: "hours".
    """

    date_time_validita: Optional[datetime] = None
    acc_time: Optional[int] = None
    acc_time_unit: Optional[str] = None
    date_time_emissione: Union[Optional[datetime], str] = "Unknown"


@dataclass
class Projection:
    """
    Dataclass per la gestione delle informazioni sulla proiezione del BUFR.
    
    -center_longitude      --Optional[float] = None   : longitudine del centro di proiezione.
    -center_latitude       --Optional[float] = None   : latitudine del centro di proiezione.
    -semimajor_axis        --Union[float,int]= 0.0    : semiasse maggiore dell'ellipsoide della proiezione.
    -semiminor_axis        --Union[float,int]= 0.0    : semiasse minore dell'ellipsoide della proiezione.
    -x_offset              --Union[float,int]= 0.0    : valore x del centro del grigliato (UTM)
                                                        (nella documentazione IDL è però indicato come
                                                        lower left corner dell'output).
    -y_offset              --Union[float,int]= 0.0    : valore y del centro del grigliato (UTM)
                                                        (nella documentazione IDL è però indicato come
                                                        lower left corner dell'output).
    -standard_par1         --Union[float,int]= 0.0    : latitudine in gradi del primo parallelo standard
                                                        su cui la scala è corretta. Il valore deve essere
                                                        compreso tra -90 e 90 gradi. NB: per proiezioni
                                                        Albers Equal Area and Lambert Conformal Conic,
                                                        standard_par1 e standard_par2 non devono avere
                                                        valori uguali e opposti (da documentazione IDL
                                                        di MAP_PROJ_INIT).
    -standard_par2         --Union[float,int]= 0.0    : latitudine in gradi del secondo parallelo standard
                                                        su cui la scala è corretta. Il valore deve essere
                                                        compreso tra -90 e 90 gradi. NB: per proiezioni
                                                        Albers Equal Area and Lambert Conformal Conic,
                                                        standard_par1 e standard_par2 non devono avere
                                                        valori uguali e opposti (da documentazione IDL
                                                        di MAP_PROJ_INIT).
    -proj_id               --Optional[int] = None     : intero identificativo della proiezione
    -proj_name             --Optional[str] = None     : nome del tipo di proiezione
    -earth_radius          --Optional[float] = None   : raggio terrestre
    -pyprojstring          --str (solo per proj_id=6,9,101)  : stringa della proiezione secondo standard di pyproj
    -zone                  --int (solo per proj_id=101): intero della zona (UTM),
                                                        compreso tra 1 e 60 per l'emisfero boreale.
    """

    proj_id: Optional[int] = None
    projection_index: Optional[int] = None
    projection_name: str = "None"
    grid_mapping_name: str = "None"
    long_name: str = "None"
    stand_par1: Optional[float] = 0.0
    semimajor_axis: Optional[float] = None
    semiminor_axis: Optional[float] = None
    x_offset: Optional[float] = None
    y_offset: Optional[float] = None
    center_latitude: Optional[float] = None
    center_longitude: Optional[float] = None
    standard_par1: Optional[float] = None
    standard_par2: Optional[float] = None
    proj_name: Optional[str] = None
    earth_radius: Optional[float] = None
    pyprojstring: Optional[str] = None
    zone: Optional[int] = None


@dataclass
class Grid:
    """
    Dataclass per la gestione delle informazioni sul grigliato del BUFR.

    -limiti               --np.ndarray = np.array([None, None, None, None])  : array numpy di 4 elementi
                                                                                che rappresenta i limiti del
                                                                                grigliato. Gli elementi sono
                                                                                ordinati come [ymin, xmin, ymax, xmax].
    -dx                   --Optional[float] = None   : passo di campionamento lungo l'asse x.
    -dy                   --Optional[float] = None   : passo di campionamento lungo l'asse y.
    -units_dx             --Optional[str] = None     : unità di misura del passo di campionamento lungo l'asse x.
    -units_dy             --Optional[str] = None     : unità di misura del passo di campionamento lungo l'asse y.
    -nx                   --Optional[int] = None      : numero di punti griglia lungo l'asse x.
    -ny                   --Optional[int] = None      : numero di punti griglia lungo l'asse y.
    """


    limiti: np.ndarray = field(default_factory=lambda: np.array([None, None, None, None]))
    dx: Optional[float] = None
    dy: Optional[float] = None
    units_dx: Optional[str] = None
    units_dy: Optional[str] = None
    nx: Optional[int] = None
    ny: Optional[int] = None


@dataclass
class OperaBufr:
    """
    Dataclass per la gestione dei dati BUFR nello standard Opera.
    https://www.eumetnet.eu/wp-content/uploads/2017/03/OPERA_2008_14_BUFR_Guidelines.pdf
    https://www.eumetnet.eu/wp-content/uploads/2017/03/ODIM-bufr-polar-and-compo-and-graphic.pdf
    https://www.eumetnet.eu/wp-content/uploads/2017/04/OPERA_BUFR_template_primer_V1.4.pdf

    -data                   --np.ndarray : array numpy con i dati del BUFR (l'image file generato da decbufr e letto usando np.fromfile(data_file, dtype='<u1'))
    -meta                   --pd.DataFrame : DataFrame pandas con i metadati del BUFR (l'out file generato da decbufr e letto pandas)
    -emission_center        --str (default="DPC") : centro di emissione dei dati. Può essere "DPC",
                                                    "Arpae Emilia-Romagna" o "Arpa Piemonte". Va specificato
                                                    per poter impostare correttamente la proiezione e interpretare
                                                    correttamente i metadati.
    -fill_method            --str (default="ave") : criterio con cui costruire la lista dei livelli
                                                    (vedi docstring di get_var_levs_from_meta).
    -source                 --Optional[str] (default=None) : sorgente dei dati. Può essere "GAT" o "SPC".
                                                             Va specificato solo se emission_center è "Arpae Emilia-Romagna".
    """

    data: np.ndarray
    meta: pd.DataFrame
    emission_center: str = "DPC"
    fill_method: str = "ave"
    source: Optional[str] = None

    def __post_init__(self):
        """
        Metodo di inizializzazione dell'oggetto OperaBufr. Si occupa di inizializzare gli attributi
        dell'oggetto e di leggere le informazioni temporali, sulla proiezione, sul grigliato e sulla variabile
        dei dati dal DataFrame pandas in cui sono storati i metadati del BUFR.
        """

        assert self.emission_center in ["DPC", "Arpae Emilia-Romagna", "Arpa Piemonte"], "emission_center non riconosciuto, scegli tra 'DPC', 'Arpae Emilia-Romagna', 'Arpa Piemonte'"
        if self.emission_center == "Arpae Emilia-Romagna":
            assert self.source in ["GAT", "SPC"], "source non riconosciuto, scegli tra 'GAT', 'SPC'"
        assert self.fill_method in ["min", "ave", "max"], "fill_method non riconosciuto, scegli tra 'min', 'ave', 'max'"
        self.time = Time()
        if self.meta.__len__() > 0:
            try:
                self.time.date_time_validita = self.get_datetime_from_meta()
                # non riempo attributi acc_time, acc_time_unit, date_time_emissione perchè non leggiamo
                # campi cumulati ma solo istantanei e non so la data di emissione
                self.time.acc_time = self.get_acc_time_from_meta()
                self.time.acc_time_unit = "hours"
            except Exception as e:
                print(f"Lettura time fallita:\n{e}")

            try:
                self.projection = self.get_projection_from_meta()
                self.grid = self.get_grid_from_meta()
                self.variable, self.levels = self.get_var_levs_from_meta()
            except Exception as e:
                print(f"Lettura projection, grid, variable fallita:\n{e}")
                print(self.meta.to_string())
                raise e
        
        self.data = self.data.reshape(self.grid.ny, self.grid.nx)

        if self.levels is not None:
            ind_lev_idl = np.arange(0, self.levels.__len__(), 1)
            # out_field = np.ones(shape=(self.grid.nx, self.grid.ny))*self.variabile.missing
            out_field = np.ones(shape=(self.grid.ny, self.grid.nx)) * self.variable.missing
            for idl in ind_lev_idl:
                # il valore della matrice in output è il livello corrispondente tra l'indice
                # del livello e il valore binario in quel pixel. Quindi se il primo elemento
                # della matrice binaria è 1, il valore in output in quel pixel è dato dal valore
                # del livello con indice 1 (self.levels[1])
                out_field[self.data == idl] = self.levels[idl]

            self.data = out_field

    def get_FXY_idx(self, F: str, X: str, Y: str, index: int = 0) -> Optional[int]:
        """
        Metodo che restituisce l'indice del DataFrame pandas in cui sono storati i metadati del BUFR
        corrispondente ai valori di F, X e Y passati come argomento.
        """
        try:
            return int(self.meta.loc[(self.meta.F == F) & (self.meta.X == X) & (self.meta.Y == Y)].index.values[index])
        except IndexError:
            return None

    def get_FXY_value(self, F: str, X: str, Y: str, index: int = 0) -> Optional[float]:
        """
        Metodo che restituisce il valore del DataFrame pandas in cui sono storati i metadati del BUFR
        corrispondente ai valori di F, X e Y passati come argomento.
        """

        try:
            return float(self.meta.loc[(self.meta.F == F) & (self.meta.X == X) & (self.meta.Y == Y)].value.values[index])
        except IndexError:
            return None
        
    def get_datetime_from_meta(self) -> Optional[datetime]:

        """
        Metodo che legge le informazioni temporali nell'oggetto DataFrame di pandas,
        in cui sono storati i metadati del BUFR, e restituisce l'oggetto datetime
        corrispondente, con la data di validità del campo.

        L'oggetto DataFrame di pandas in cui sono storati i metadati del file BUFR
        è ottenuto in output dal metodo simcradarlib.io_utils.bufr_class.read_meta().
        Quando viene invocato il metodo di istanza
        simcradarlib.io_utils.bufr_class.read_bufr() per leggere il BUFR, tale
        DataFrame viene assegnato all'attributo di istanza 'meta'.
        (Per questo nel codice si accede ad esso tramite self.meta)

        OUTPUT:
        - dt            --datetime : oggetto datetime.datetime contenente la dataora di
                                     validità dei dati nel file BUFR.
        """

        year, month, day, hour, minutes = (-99, -99, -99, -99, -99)
        try:
            index_year = self.meta.loc[
                (self.meta.F == "3") & (self.meta.X == "1") & (self.meta.Y == "11")
            ].index.values[0]
            year = int(float(self.meta.iloc[index_year].value))
        except IndexError:
            print("Non trovo YEAR (F=3,X=1,Y=11)")

        try:
            subdf = self.meta.iloc[index_year + 1]
            if subdf.F is None and subdf.X is None and subdf.Y is None:
                month = int(float(subdf.value))
            subdf = self.meta.iloc[index_year + 2]
            if subdf.F is None and subdf.X is None and subdf.Y is None:
                day = int(float(subdf.value))
        except IndexError:
            print("Mese e giorno non estratti")

        try:
            index_hour = self.meta.loc[
                (self.meta.F == "3") & (self.meta.X == "1") & (self.meta.Y == "12")
            ].index.values[0]
            hour = int(float(self.meta.iloc[index_hour].value))
            subdf = self.meta.iloc[index_hour + 1]
            if subdf.F is None and subdf.X is None and subdf.Y is None:
                minutes = int(float(subdf.value))
        except IndexError:
            print("non trovo campi 'hour,minutes'")

        try:
            year = year if year != -99.0 else 0
            month = month if month != -99.0 else 0
            day = day if day != -99.0 else 0
            hour = hour if hour != -99.0 else 0
            minutes = minutes if minutes != -99.0 else 0

            dt = datetime(year, month, day, hour, minutes)
        except ValueError:
            print("non sono stati letti i campi Y,m,d,H,M")
            dt = None
        return dt

    def get_acc_time_from_meta(self) -> int:

        """
        Metodo che legge nell'oggetto DataFrame di pandas,
        in cui sono storati i metadati del file BUFR,
        il tempo di cumulata se il campo è una cumulata.
        Se il campo è istantaneo o la lettura di acc_time fallisce,
        restituisce l'intero acc_time=0 di default.

        L'oggetto DataFrame di pandas in cui sono storati i metadati del file BUFR
        è ottenuto in output dal metodo simcradarlib.io_utils.bufr_class.read_meta().
        Quando viene invocato il metodo di istanza
        simcradarlib.io_utils.bufr_class.read_bufr() per leggere il BUFR, tale
        DataFrame viene assegnato all'attributo di istanza 'meta'.
        (Per questo nel codice si accede ad esso tramite self.meta)

        OUTPUT:
        acc_time        --int : intero corrispondente al tempo di cumulata in ore se il
                                campo è cumulato, 0 altrimenti.

        """

        acc_time = 0
        if self.get_FXY_value("0", "8", "21") == 3.0:
            t = self.get_FXY_value("0", "4", "23")
            if t is not None:
                acc_time -= int(24 * t)
            else:
                print("Non trovo time period per descrittori F=0,X=4,Y=23")

            t2 = self.get_FXY_value("0", "4", "24")
            if t2 is not None:
                acc_time -= int(t2)
            else:
                print("Non trovo time period per descrittori F=0,X=4,Y=24")

            return int(acc_time)

    def get_projection_from_meta(self) -> Projection:

        """
        Metodo che legge i dati sulla proiezione dal DataFrame pandas in cui sono storati i metadati
        del BUFR e crea l'istanza della classe Projection.

        L'oggetto DataFrame di pandas in cui sono storati i metadati del file BUFR è ottenuto in output
        dal metodo simcradarlib.io_utils.bufr_class.read_meta().
        Quando viene invocato il metodo di istanza simcradarlib.io_utils.bufr_class.read_bufr() per leggere il
        BUFR, tale DataFrame viene assegnato all'attributo di istanza 'meta' (per questo nel codice si accede
        ad esso tramite self.meta).

        OUTPUT:
        - proj_struct    --StructProjection :
                                      istanza della classe Projection, avente attributi:

                                      center_longitude   --Optional[float] = None :
                                                             longitudine del centro di proiezione.
                                      center_latitude    --Optional[float] = None :
                                                             latitudine del centro di proiezione.
                                      semimajor_axis     --Union[float,int]= 0.0  :
                                                             semiasse maggiore dell'ellipsoide della proiezione.
                                      semiminor_axis     --Union[float,int]= 0.0  :
                                                             semiasse minore dell'ellipsoide della proiezione.
                                      x_offset           --Union[float,int]= 0.0  :
                                                             valore x del centro del grigliato (UTM)
                                                             (nella documentazione IDL è però indicato come
                                                             lower left corner dell'output).
                                      y_offset           --Union[float,int]= 0.0  :
                                                             valore y del centro del grigliato (UTM)
                                                             (nella documentazione IDL è però indicato come
                                                             lower left corner dell'output).
                                      standard_par1      --Union[float,int]= 0.0  :
                                                             latitudine in gradi del primo parallelo standard
                                                             su cui la scala è corretta. Il valore deve essere
                                                             compreso tra -90 e 90 gradi. NB: per proiezioni
                                                             Albers Equal Area and Lambert Conformal Conic,
                                                             standard_par1 e standard_par2 non devono avere
                                                             valori uguali e opposti (da documentazione IDL
                                                             di MAP_PROJ_INIT).
                                      standard_par2      --Union[float,int]= 0.0  :
                                                             latitudine in gradi del secondo parallelo standard
                                                             su cui la scala è corretta. Il valore deve essere
                                                             compreso tra -90 e 90 gradi. NB: per proiezioni
                                                             Albers Equal Area and Lambert Conformal Conic,
                                                             standard_par1 e standard_par2 non devono avere
                                                             valori uguali e opposti (da documentazione IDL
                                                             di MAP_PROJ_INIT).
                                      proj_id            --Optional[int] = None   :
                                                             intero identificativo della proiezione
                                      proj_name          --Optional[str] = None   :
                                                             nome del tipo di proiezione
                                      earth_radius       --Optional[float] = None :
                                                             raggio terrestre
                                      pyprojstring       --str (solo per proj_id=6,9,101)  :
                                                             stringa della proiezione secondo standard di pyproj
                                      zone               --int (solo per proj_id=101): intero della zona (UTM),
                                                             compreso tra 1 e 60 per l'emisfero boreale.

        Se la lettura fallisce viene restituita l'istanza con attributi di classe settati ai valori di default.

        Differenze rispetto a IDL:
        1.  Per attinenza alle convenzioni Python PEP 8 sul name styling
            (https://peps.python.org/pep-0008/#naming-conventions), alcune variabili sono
            state rese 'lowercase' in questa implementazione, mentre in IDL erano 'UPPER_CASE'.
            Tali variabili sono:
            SEMIMAJOR_AXIS, SEMIMINOR_AXIS, STANDARD_PAR1, STANDARD_PAR2, ZONE.
        """

        proj_struct = Projection()
        proj_struct.center_longitude = self.get_FXY_value("0", "29", "193")
        proj_struct.center_latitude = self.get_FXY_value("0", "29", "194")
        proj_struct.semimajor_axis = self.get_FXY_value("0", "29", "199") or 0.0
        proj_struct.semiminor_axis = self.get_FXY_value("0", "29", "200") or 0.0
        proj_struct.x_offset = self.get_FXY_value("0", "29", "195") or 0.0
        if self.emission_center == "DPC" and proj_struct.x_offset > 0:
            print("x_offset positivo, lo rendo negativo per DPC")
            proj_struct.x_offset = -proj_struct.x_offset
        proj_struct.y_offset = self.get_FXY_value("0", "29", "196") or 0.0
        proj_struct.standard_par1 = self.get_FXY_value("0", "29", "197") or 0.0
        proj_struct.standard_par2 = self.get_FXY_value("0", "29", "198") or 0.0

        try:
            # qui potrei anzichè usare  f,x,y=0,29,201 o 1 , potrei invece filtrare su descriptor='Projection type'
            index_ = self.meta.loc[(self.meta.F == "0") & (self.meta.X == "29") & (self.meta.Y == "201")].index.values
            if index_.__len__() > 0:
                index_ = index_[0]
            else:
                index_ = self.meta.loc[
                    (self.meta.F == "0") & (self.meta.X == "29") & (self.meta.Y == "1")
                ].index.values[0]

            proj_type_int = int(float(self.meta.iloc[index_].value))
            if proj_type_int == 0:
                # 0: Azimuthal Equidistant
                # proj_struct.proj_id = 6 # tolgo proj_id residuo idl
                if self.emission_center == "Arpae Emilia-Romagna":
                    proj_struct.projection_index = 0
                    if self.source == "GAT":
                        proj_struct.center_longitude = 10.4992
                        proj_struct.center_latitude = 44.7914
                    elif self.source == "SPC":
                        proj_struct.center_longitude = 11.6236
                        proj_struct.center_latitude = 44.6547
                    # aggiungo proj_name,Earth radius come per zlr
                    # aggiungo anche pyprojstring
                    proj_struct.proj_name = "Cartesian lat-lon"
                    proj_struct.earth_radius = 6370.997
                    pyprojstring = f"+proj=eqc +lat_0={proj_struct.center_latitude:.4f} \
    +lon_0={proj_struct.center_longitude:.4f} +ellps=WGS84 +R={proj_struct.earth_radius:.4f}"
                    proj_struct.pyprojstring = pyprojstring

            elif proj_type_int == 1:
                # 1 stereographic - 106 polar stereographic
                if proj_struct.center_latitude == 90.0:
                    # stereografica polare (idl)
                    # proj_struct.proj_id = 106 # tolgo proj_id residuo idl
                    proj_struct.proj_name = "Polar stereographic"
                    proj_struct.center_latitude = proj_struct.stand_par1
                else:
                    # stiamo utilizzando la stereografica normale non devo ridefinire i parametri
                    # proj_struct.proj_id = 1 # tolgo proj_id residuo idl
                    proj_struct.proj_name = "Stereographic"
            elif proj_type_int == 2:
                # proj_struct.proj_id = 104  # lambert conical # tolgo proj_id residuo idl
                proj_struct.proj_name = "Lambert Conformal Conic"
                # proj_struct.addparams(["STANDARD_PAR1","STANDARD_PAR2"],[stand_par1,stand_par2]) gia fatto
            elif proj_type_int == 3:
                # salto la parte che entra nel if keyword set = DPC
                if self.emission_center == "DPC":
                    # proj_struct.proj_id = 9 # tolgo proj_id residuo idl
                    # per il DPC utilizzo la proiezione mercatore e
                    # impongo il raggio della sfera visto che DATAMAT ha utilizzato questi parametri
                    if (proj_struct.semimajor_axis, proj_struct.semiminor_axis) == (0.0, 0.0):
                        proj_struct.semimajor_axis = 6370997.0
                        proj_struct.semiminor_axis = 6370997.0
                        proj_struct.earth_radius = 6370997.0
                    else:
                        proj_struct.earth_radius = proj_struct.semimajor_axis
                        # pyprojstring='+proj=tmerc +lat_0=42.0 +lon_0=12.5 +ellps=WGS84'
                    pyprojstring = f"+proj=tmerc +lat_0={proj_struct.center_latitude:.1f} \
    +lon_0={proj_struct.center_longitude:.1f} +ellps=WGS84"
                    proj_struct.pyprojstring = pyprojstring
                    proj_struct.proj_name = "tmerc"  # =pyproj.Proj(pyprojstring).name

                elif self.emission_center == "Arpa Piemonte":
                    # piemonte codifica in UTM:
                    # proj_struct.proj_id = 101 #in lettura togliamo residui di idl
                    proj_struct.earth_radius = 6378388.0
                    proj_struct.proj_name = "UTM"
                    proj_struct.semimajor_axis = 6378388.0
                    proj_struct.semiminor_axis = 6356911.94613
                    proj_struct.zone = 32

                    # per il piemonte calcolo anche parametri x_offset e y_offset che non vengono passati
                    pyprojstring = "+proj=utm +zone=32 +k_0=0.9996 +ellps=intl"
                    piem_proj = pyproj.Proj(pyprojstring)
                    proj_struct.pyprojstring = pyprojstring
                    proj_struct.proj_name = piem_proj.name  # = 'utm'

                    # oppure posso scrivere proj_name'utm intl' (International 1909 (Hayford) )
                    index_ = self.meta.loc[
                        (self.meta.F == "3") & (self.meta.X == "1") & (self.meta.Y == "23")
                    ].index.values[0]
                    lat_nw = float(self.meta.iloc[index_].value)
                    lon_nw = round(float(self.meta.iloc[index_ + 1].value), 1)
                    xoffset, yoffset = piem_proj(lon_nw, lat_nw)
                    proj_struct.x_offset = xoffset
                    proj_struct.y_offset = yoffset

                else:
                    # proj_struct.proj_id = 9 # tolgo proj_id residuo idl
                    proj_struct.proj_name = "Mercator"
            elif proj_type_int == 4:
                # proj_struct.proj_id = 6 # tolgo proj_id residuo idl
                proj_struct.proj_name = "Azimuthal Equidistant"
            elif proj_type_int == 5:
                # proj_struct.proj_id = 4 # tolgo proj_id residuo idl
                proj_struct.proj_name = "Lambert Azimuthal"
            else:
                print("proj_type_int non noto")

        except IndexError:
            print("non leggo descriptor 'Projection type'")

        except AttributeError:
            raise RuntimeError("tentativo di accesso ad attributo inesistente")

        return proj_struct

    def get_grid_from_meta(self) -> Grid:

        """
        Metodo che legge le informazioni sul grigliato dei dati BUFR dall'oggetto
        DataFrame pandas in cui sono storati e crea istanza di classe Grid.

        L'oggetto DataFrame di pandas in cui sono storati i metadati del file BUFR
        è ottenuto in output dal metodo simcradarlib.io_utils.bufr_class.read_meta().
        Quando viene invocato il metodo di istanza
        simcradarlib.io_utils.bufr_class.read_bufr() per leggere il BUFR, tale
        DataFrame viene assegnato all'attributo di istanza 'meta'.
        (Per questo nel codice si accede ad esso tramite self.meta)

        OUTPUT:
        - grid_struct      --Grid : istanza della classe Grid.
        Differenze rispetto a IDL:
        In questa implementazione sono stati aggiunti gli attributi 'units_dx' e 'units_dy'
        in quanto attributi di istanza della classe Grid, che implementa la
        struttura con le informazioni sul grigliato.
        """
        ny = int(self.get_FXY_value("0", "30", "22"))
        nx = int(self.get_FXY_value("0", "30", "21"))
        dy = self.get_FXY_value("0", "6", "33")
        dx = self.get_FXY_value("0", "5", "33")
        units_dx = "meters"
        units_dy = "meters"
        x_offset, y_offset = (self.projection.x_offset, self.projection.y_offset)
        limiti = np.array(
            [y_offset - ny * dy, x_offset, y_offset, x_offset + nx * dx]
        )

        grid_struct = Grid(limiti=limiti, dx=dx, dy=dy, units_dx=units_dx, units_dy=units_dy, nx=nx, ny=ny)

        return grid_struct

    def get_var_levs_from_meta(self) -> Tuple[Variable, np.array]:

        """
        Metodo che legge informazioni sulla variabile dei dati nel BUFR dal DataFrame in cui sono storati
        e crea un'istanza della classe figlia di Variable implementata nel modulo variabili.py

        Viene inoltre ricavato l'array dei livelli, cioè le classi di valori possibili per la variabile.
        Infatti nei BUFR il campo dei dati non è trattato come un campo continuo ma come categorico,
        cioè si assume che il campo ammette nei singoli punti griglia un insieme finito di valori.

        I livelli vengono utilizzati nell'estrazione dei valori del campo di dati successivamente.

        L'oggetto DataFrame di pandas in cui sono storati i metadati del file BUFR è ottenuto in output
        dal metodo simcradarlib.io_utils.bufr_class.read_meta().
        Quando viene invocato il metodo di istanza simcradarlib.io_utils.bufr_class.read_bufr() per leggere
        il BUFR, tale DataFrame viene assegnato all'attributo di istanza 'meta'(per questo nel codice si
        accede ad esso tramite self.meta).

        ______________________________________Approfondimento:____________________________________________

        Questo approfondimento serve solo per avere un'idea di come sono i dati del campo in un BUFR.
        Dal file BUFR vengono letti i livelli con i quali si individua una sequenza :
        ad esempio se i livelli letti sono [l0, l1, l2, l3 ] allora si considera che l'asse dei valori
        è dato da:
                        l0               l1                 l2            l3
                         |_______________|__________________|_____________|

        che equivale a individuare 3 classi di valori, con estremi [l0,l1], [l1,l2], [l2,l3] rispettivamente.
        L'insieme finale di valori ammessi per il campo (chiamati anch'essi livelli, sarebbero i veri
        livelli finali) è ottenuto prendendo il minimo per ogni classe se fill_method='min', il massimo su
        ogni classe se fill_method='max' e la media degli estremi se fill_method='ave'.
        In questo esempio, se fill_method è
        - 'min' ---> livelli finali sono [l0,l1,l2]
        - 'max' ---> livelli finali sono [l1,l2,l3]
        - 'ave' ---> livelli finali sono [(l0+l1)/2,(l1+l2)/2,(l2+l3)/2]
        In realtà in questo insieme finale di livelli, il primo livello è sempre il 'lower_bound' letto
        dal file BUFR.
        __________________________________________________________________________________________________

        INPUT:
        - fill_method        --str: indica il criterio con cui costruire la lista dei livelli
                                    -'min' se i livelli sono definiti sul bottom della classe
                                    -'ave' se i livelli sono la media tra gli estremi della classe
                                    -'max' se i livelli sono definiti sul top della classe
                                    Il primo livello è sempre dato dal lower bound e viene letto dal BUFR.

        OUTPUT:
        - ( struct_var, levels):    --Tuple : tupla di due elementi che sono
                                              - struct_var   --Union[StructVariable,VarZ,VarPrmm,VarCumPrmm] :
                                                                    classe figlia di variabile che implementa
                                                                    il tipo di variabile dei dati BUFR.
                                              - levels       --np.array[float] :
                                                                    array dei livelli che identificano le classi
                                                                    di valori possibili per la variabile.
        """

        # le tabelle delle variabili radar usate in idl sono in
        # /autofs/radar/radar/file_x_idl/tabelle/

        # cerco la riga che contiene la variabile radar 
        index_z = self.get_FXY_idx("3", "13", "9") # Horizontal reflectivity
        index_rr = self.get_FXY_idx("3", "13", "10") # Radar rainfall intensity
        index_cum = self.get_FXY_idx("3", "13", "11") 
        index_cum2 = self.get_FXY_idx("0", "13", "11") # Total precipitation/total water equivalent

        if index_z is not None:
            var = VarZ()
            index_ = index_z
        elif index_rr is not None:
            var = VarPrmm()
            index_ = index_rr
        elif index_cum is not None or index_cum2 is not None:
            var = VarCumPrmm()
            index_ = index_cum if index_cum is not None else index_cum2
        else:
            raise Exception("Non ho capito cosa contiene il bufr. Esco")

        index_ = int(index_)

        # leggo anche i livelli
        low_bounds = float(self.meta.iloc[index_].value)
        if self.emission_center == "DPC" and var.name == "Z" and low_bounds == 0:
            low_bounds = -31  # patch per parare buco di Datamat
            print("patch per parare buco di Datamat")
        if (self.meta.iloc[index_ + 1].F, self.meta.iloc[index_ + 1].X, self.meta.iloc[index_ + 1].Y) == (
            "1",
            "1",
            "0",
        ):
            index_ = index_ + 1

        # il numero di livelli è scritto nella riga dopo il primo record con descriptor=var.name
        nlev = max(0, int(float(self.meta.iloc[index_ + 1].value)))

        # prendo tutti i livelli in ordine crescente e se un livello è 'missing' non lo considero
        ind_levs = np.array(range(index_ + 2, index_ + 2 + nlev))
        levels = self.meta.iloc[ind_levs].value.values
        levels = levels[levels != "missing"].astype(float)
        levels = levels.tolist()
        
        # aggiungo il lower bound come primo livello
        levels.insert(0, low_bounds)
        
        # levels deve essere monotono crescente e con i valori tutti maggiori o uguali a low_bounds
        assert np.all(np.diff(levels) >= 0), "Livelli non sono in ordine crescente"
        
        if self.fill_method == "min":
            # livelli sono definiti sul bottom della classe
            levels[1:] = levels[0:-1]
        elif self.fill_method == "ave":
            levels[1:] = np.array([np.mean(levels[i: i + 2]) for i in range(levels.__len__() - 1)])
        elif self.fill_method == "max":
            # levels[1:] = levels[1:] passo tanto non fa niente
            pass
        else:
            raise ValueError("Parametro fill_method passato in modo scorretto. Esco")

        return var, levels

    def _metadata_to_dict(self) -> dict[str, Any]:
        return {
            "variable_name": self.variable.name,
            "units": self.variable.units,
            "fill_method": self.fill_method,
            "variable_standard_name": self.variable.standard_name,
            "variable_long_name": self.variable.long_name,
            "min_value": self.variable.min_val,
            "max_value": self.variable.max_val,
            "undetect": self.variable.undetect,
            "datetime": self.time.date_time_validita.strftime("%Y-%m-%d %H:%M:%S"),
            "emission_center": self.emission_center,
            "source": self.source,
            "projection": self.projection.proj_name,
            "center_longitude": self.projection.center_longitude,
            "center_latitude": self.projection.center_latitude,
            "semimajor_axis": self.projection.semimajor_axis,
            "semiminor_axis": self.projection.semiminor_axis,
            "x_offset": self.projection.x_offset,
            "y_offset": self.projection.y_offset,
            "standard_par1": self.projection.standard_par1,
            "standard_par2": self.projection.standard_par2,
            "pyprojstring": self.projection.pyprojstring,
            "grid_nx": self.grid.nx,
            "grid_ny": self.grid.ny,
            "grid_dx": self.grid.dx,
            "grid_dy": self.grid.dy,
            "grid_units_dx": self.grid.units_dx,
            "grid_units_dy": self.grid.units_dy,
            "grid_xmin": self.grid.limiti[1],
            "grid_xmax": self.grid.limiti[3],
            "grid_ymin": self.grid.limiti[0],
            "grid_ymax": self.grid.limiti[2],
        }

    def to_geotiff(self, path: Optional[str] = None) -> Optional[bytes]:

        """
        Metodo per esportare i dati del BUFR in un file GeoTiff georeferenziato.

        INPUT:
        - path          --str : percorso del file GeoTiff in cui esportare i dati.
                                Se non specificato, il file viene creato in memoria.
    
        OUTPUT:
        - Optional[bytes] : se path=None, restituisce i byte del file GeoTiff creato in memoria.
        """

        try:
            import rasterio
        except ImportError:
            raise ImportError("Il modulo rasterio non è installato. Impossibile esportare in GeoTiff.")

        # Create a memory file if no path is provided
        output_file = path if path is not None else rasterio.io.MemoryFile()


        # Create the output GeoTiff
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=self.grid.ny,
            width=self.grid.nx,
            count=1,
            dtype=rasterio.float32,
            crs=rasterio.crs.CRS.from_proj4(self.projection.pyprojstring),
            transform=rasterio.transform.from_origin(
                self.grid.limiti[1], self.grid.limiti[2], self.grid.dx, self.grid.dy
            ),
            nodata=self.variable.missing,
            compress="ZSTD",
            predictor=2,
            tiled=True,
        ) as dst:
            dst.write(self.data.astype(rasterio.float32), 1)
            dst.update_tags(**self._metadata_to_dict())

        if path is None:
            output_file.seek(0)
            return output_file.read()

    def to_netcdf(self, path: Optional[str] = None) -> Optional[bytes]:
            
            """
            Metodo per esportare i dati del BUFR in un file NetCDF.
    
            INPUT:
            - path          --str : percorso del file NetCDF in cui esportare i dati.
                                    Se non specificato, il file viene creato in memoria.
        
            OUTPUT:
            - Optional[bytes] : se path=None, restituisce i byte del file NetCDF creato in memoria.
            """
    
            try:
                import netCDF4
            except ImportError:
                raise ImportError("Il modulo netCDF4 non è installato. Impossibile esportare in NetCDF.")
            
            try:
                import rasterio
            except ImportError:
                raise ImportError("Il modulo rasterio non è installato. Impossibile esportare in NetCDF.")

            # Create a memory file if no path is provided
            output_file = path if path is not None else io.BytesIO()
    
            # Create the output NetCDF
            with netCDF4.Dataset(output_file, "w") as dst:

                dst.createDimension("x", self.grid.nx)
                dst.createDimension("y", self.grid.ny)
                dst.createDimension("time", 1)
                dst.createDimension("spatial_ref", 1)

                dst.createVariable("x", "f4", ("x",))
                dst["x"][:] = np.linspace(self.grid.limiti[1], self.grid.limiti[3]-self.grid.dx, self.grid.nx)
                dst["x"].setncatts({
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate of projection",
                    "units": "m",
                    "axis": "X",
                })

                dst.createVariable("y", "f4", ("y",))
                dst["y"][:] = np.linspace(self.grid.limiti[0], self.grid.limiti[2]-self.grid.dy, self.grid.ny)
                dst["y"].setncatts({
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate of projection",
                    "units": "m",
                    "axis": "Y",
                })

                dst.createVariable("time", "f8", ("time",))
                dst["time"][:] = netCDF4.date2num([self.time.date_time_validita], units="seconds since 1970-01-01", calendar="standard")
                dst["time"].setncatts({
                    "standard_name": "time",
                    "long_name": "time",
                    "calendar": "gregorian",
                    "units": "seconds since 1970-01-01 00:00:00",
                    "axis": "T",
                })

                dst.createVariable("spatial_ref", "c")
                wkt = rasterio.crs.CRS.from_proj4(self.projection.pyprojstring).wkt
                dst["spatial_ref"].setncatts({
                    "spatial_ref": wkt,
                })

                dst.createVariable(self.variable.name, "f4", ("time", "y", "x"), fill_value=self.variable.missing, zlib=True, complevel=9)
                dst[self.variable.name][:] = np.flipud(self.data)[np.newaxis, ...]
                
                meta_dict = self._metadata_to_dict()
                meta_dict = {}
                meta_dict['grid_mapping'] = 'spatial_ref'
                dst[self.variable.name].setncatts(meta_dict)

                dst.setncatts({
                    "Conventions": "CF-1.7",
                    "title": f"{self.variable.long_name} ({self.variable.units})",
                    "source": self.source,
                    "emission_center": self.emission_center,
                    "fill_method": self.fill_method,
                })
    
            if path is None:
                output_file.seek(0)
                return output_file.read()

    def cartopy_plot(self,
                     cmap: Optional[Any] = None,
                     norm: Optional[Any] = None,
                     title: Optional[str] = None,
                     use_projection: bool = False,
                     features: Optional[Any] = None,
                     gridlines: bool = True,
                     figsize: Tuple[int, int] = (10, 10),
                     dpi: int = 100,
                     **kwargs,
                     ):
        """
        Metodo per visualizzare i dati del BUFR su una mappa utilizzando Cartopy.

        INPUT:
        - cmap            --Any : scala di colori da utilizzare per il plot. Se None, viene utilizzata la scala
                                    di default.
                                    Default: None.
        - norm            --Any : norma da utilizzare per il plot. Se None, viene utilizzata la norma di default.
                                    Default: None.
        - title           --str : titolo del plot.
                                    Default: None.
        - use_projection  --bool: se True, utilizza la proiezione specificata nel BUFR, altrimenti utilizza
                                    la proiezione PlateCarree.
                                    Default: False.
        - features        --Any : lista di feature da aggiungere alla mappa.
                                    Default: [cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAND, cfeature.OCEAN].
        - gridlines       --bool: se True, aggiunge le griglie alla mappa.
                                    Default: True.
        - figsize         --Tuple[int,int]: dimensioni del plot.
                                    Default: (10, 10).
        - dpi            --int : risoluzione del plot.
                                    Default: 100.
        - **kwargs        --dict: argomenti aggiuntivi da passare alla funzione plt.contourf().
        """

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import matplotlib.pyplot as plt
            import matplotlib
            
        except ImportError:
            raise ImportError("I moduli cartopy e matplotlib non sono installati. Impossibile visualizzare il plot.")

        # create the grid of coordinates
        xs = np.linspace(self.grid.limiti[1],self.grid.limiti[3],int(self.grid.nx))
        ys = np.linspace(self.grid.limiti[0],self.grid.limiti[2],int(self.grid.ny))
        xs,ys = np.meshgrid(xs,ys)

        # read and apply projection from meta
        if self.projection.proj_name == "tmerc" and use_projection:
            proj = ccrs.TransverseMercator(
                central_longitude=self.projection.center_longitude,
                central_latitude=self.projection.center_latitude,
                globe=ccrs.Globe(
                    ellipse="WGS84",
                    semimajor_axis=self.projection.semimajor_axis,
                    semiminor_axis=self.projection.semiminor_axis,
                    )
                )
        elif self.projection.proj_name == "utm" and use_projection:
            proj = ccrs.UTM(
                zone=self.projection.zone,
                globe=ccrs.Globe(ellipse="WGS84", semimajor_axis=self.projection.semimajor_axis),
            )
        else:
            proj = ccrs.PlateCarree()
            original_proj = pyproj.Proj(self.projection.pyprojstring)
            xs,ys = original_proj(xs,ys, inverse=True)

        # Create the figure and axis
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=proj)

        features = [cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAND, cfeature.OCEAN] if features is None else features
        for feature in features:
            ax.add_feature(feature)

        # Add gridlines
        if gridlines:
            gl=ax.gridlines(draw_labels=True,linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabels_top = False
            gl.ylabels_right = False

        # Set the colormap and norm
        if cmap is None:
            if self.variable.units == "mm":
                cmap, norm, _, _ = get_colormap("depth", units="mm", colorscale="STEPS-BE")
            elif self.variable.units == "mm h-1":
                cmap, norm, _, _ = get_colormap("intensity", units="mm/h", colorscale="STEPS-BE")
            elif self.variable.units == "dBZ":
                cmap, norm, _, _ = get_colormap("intensity", units="dBZ", colorscale="STEPS-BE")
            else:
                cmap = cmap
                norm = norm
        else:
            cmap = cmap
            norm = norm

        # Plot the data
        data_plot = np.flipud(self.data.copy())
        data_plot[data_plot == self.variable.missing] = np.nan
        im = ax.pcolormesh(xs, ys, data_plot, transform=proj, cmap=cmap, norm=norm, **kwargs)

        # Add the title
        if title is None:
            dt_string = self.time.date_time_validita.strftime("%Y-%m-%d %H:%M UTC")
            entity = self.emission_center
            title = f"{entity} - {dt_string}\n{self.variable.long_name} ({self.variable.units})"
        plt.title(title)

        # Add the colorbar
        cax,kw=matplotlib.colorbar.make_axes([ax],location='right',pad=0.02,shrink=0.5)
        cbar=fig.colorbar(im,cax=cax,extend='max',**kw)
        cbar.set_label(self.variable.units,labelpad=-20,y=1.10,rotation=0)
        # cbar.set_label(f"{self.variable.long_name} ({self.variable.units})")

        return fig, ax, cbar


class Decbufr:
    """
    Wrapper per l'eseguibile decbufr, che permette di processare i file BUFR e ottenere i dati e i metadati.
    L'eseguibile decbufr è un software sviluppato da EUMETNET per la decodifica dei file BUFR OPERA.
    Questro wrapper esgue come sotto-processo l'eseguibile decbufr e legge i file di output per ottenere i dati e i metadati.

    Esempio di utilizzo:
    >>> decbufr = DecbufrWrapper()
    >>> opera_bufr = decbufr.process_bufr('path/to/bufr/file')
    ...
    >>> del decbufr


    -decbufr_path       --str : percorso dell'eseguibile decbufr o del file tar.gz contenente l'eseguibile e le tabelle BUFR.
                                  Se non specificato, viene utilizzato il file decbufr.tgz presente nella stessa directory del modulo
                                  che viene estratto in una directory temporanea.
    """

    def __init__(self, decbufr_path: Optional[str] = None):
        # Determine the default path where the tar.gz file or executable might be located
        default_path = Path(__file__).parent / 'decbufr.tgz'

        # Check if path is provided or not
        if decbufr_path is None:
            decbufr_path = default_path

        self.executable_path = None

        # Check the file type using magic
        if os.path.isdir(decbufr_path):
            # If provided path is a directory, assume it's the directory containing the executable
            self.executable_path = os.path.join(decbufr_path, 'decbufr')
        else:
            file_type = magic.from_file(decbufr_path, mime=True).split('/')[-1]

            if 'gzip' in file_type:
                # If the path is a tar.gz file, extract it to a temporary directory
                self.temp_dir = tempfile.TemporaryDirectory()
                with tarfile.open(decbufr_path, 'r:gz') as tar_ref:
                    extractall_kwargs = {'filter': 'fully_trusted'} if sys.version_info >= (3, 12) else {}
                    tar_ref.extractall(self.temp_dir.name, **extractall_kwargs)
                self.executable_path = os.path.join(self.temp_dir.name, 'decbufr')
            elif 'executable' in file_type or 'octet-stream' in file_type:
                # If provided path is an executable file
                self.executable_path = decbufr_path
            else:
                raise ValueError(f"The provided path '{decbufr_path}' is neither a tar.gz file, a directory, nor an executable file. ({file_type})")

        # Check if the executable exists at the determined path
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(f"The executable '{self.executable_path}' does not exist.")

        # Final check: try to run the executable
        try:
            result = subprocess.run([self.executable_path], capture_output=True, text=True)
            if result.returncode != 0 and 'Usage:' not in result.stderr:
                raise RuntimeError(f"The executable '{self.executable_path}' could not be run. Error: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Failed to execute '{self.executable_path}': {e}")

    def read_bufr(self, bufr_file: str, tabdir: str = None, fill_method: str = "ave") -> OperaBufr:
        """
        Processa un file BUFR e restituisce un oggetto OperaBURF contenente i dati e i metadati.

        INPUT:
        - bufr_file         --str : percorso del file BUFR da processare.
        - tabdir            --str : percorso della directory contenente le tabelle BUFR.
                                      Se non specificato, viene utilizzata la directory temporanea creata durante l'inizializzazione.
        - fill_method       --str : metodo di riempimento dei livelli mancanti.
                                      I valori possibili sono 'min', 'ave', 'max'.
                                      Default: 'ave'.
        """

        # Create temporary files for output and image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_output:
            metadata_file = temp_output.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.data') as temp_image:
            data_file = temp_image.name

        try:
            # Run the decbufr command 
            command = [self.executable_path]
            
            if tabdir:
                command.extend(['-d', tabdir])
            elif hasattr(self, 'temp_dir'):
                command.extend(['-d', self.temp_dir.name])
            else:
                raise ValueError("The path to the directory containing the BUFR tables (localtabb...csv and bufrtab...csv) is required.")
            
            command.extend([bufr_file, metadata_file, data_file])
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with error: {result.stderr}")

            # Read the output text file
            meta = self.read_meta(metadata_file)

            # Read the data array
            # the data is stored as unsigned 8-bit integers
            data = np.fromfile(data_file, dtype='<u1')

        finally:
            # Delete temporary files
            os.remove(metadata_file)
            os.remove(data_file)

        # infer emission center and source from filename
        try:
            namefile = os.path.basename(bufr_file)
            if "EMRO" in namefile:
                name_source = namefile.split("_")[3].strip("@")
                emission_center = "Arpae Emilia-Romagna"
                # prod = namefile.split("_")[4].strip("@")
                # prod = "LBM" if prod == "CZ" else prod
            elif "PIEM" in namefile:
                name_source = "Bric della Croce,Monte Settepani"  # o TORINO? me lo invento
                emission_center = "Arpa Piemonte"
                # prod = namefile.split("_")[4].strip("@")
            elif "ROMA" in namefile:
                name_source = "Mosaico radar nazionale"  # me lo invento
                emission_center = "DPC"
                # prod = namefile.split("_")[2].strip("@")
            else:
                emission_center = "DPC"
                name_source = "Sorgente non riconosciuta"
        except IndexError:
            emission_center = "DPC"
            name_source = "Sorgente non riconosciuta"

        return OperaBufr(data, meta, emission_center=emission_center, source=name_source, fill_method=fill_method)
    
    def _check_emptyfield(self, substring: str) -> Optional[str]:

        """
        Metodo che restituisce None se la stringa in input, substring,
        è vuota, altrimenti restituisce substring.

        INPUT:
        - substring   --str : stringa in input.
        OUTPUT:
                      --Union[str,None]: substring se subtring non vuota
                                         altrimenti None.

        Questo metodo è invocato dal metodo di istanza della classe Bufr get_octect()
        per ricostruire gli ottetti nel file di metadati di un BUFR in lettura.
        """

        if substring == "":
            return None
        else:
            return substring
    
    def get_octect(self, line: str, col_ends: list[int]) -> list[str]:

        """
        Metodo che prende in input una riga del file testo dei metadati
        ottenuto dal decbufr (nomebufr.txt) e crea la lista di 8 componenti
        relative ai metadati. Le componenti che corrispondono a campi vuoti vengono
        settati a None tramite il metodo _check_emptyfield (sopra)

        INPUT:
        - line           --str : riga del file di metadati in output alla decodifica
                                 del bufr tramite script C decbufr.
                                 Tale riga è una stringa, contenente 8 componenti
                                 (sottostringhe) sui metadati.
        OUTPUT:
        - octect         --list : lista avente le seguenti 8 componenti
                                  [f_, x_, y_, value, id5, id6, id7, descriptor]

        dove f_,x_,y_ rappresentano il descrittore BUFR, che può essere di diversi tipi:
        (***********************inizio source documentazione Mistral***************************)

        - Element descriptor: f_=0, x_=classe del descrittore, y_=entry within a class x_
                              (Tabella B)
        - Replica di numero predefinito di descrittori:
                               f_=1, x_=numero di descrittori da replicare, y_=numero di repliche
                               (y_=0 guarda documentazione)
        - Operator descriptor: f_=2, x_=identifica operatore, y_=valore per controllo sull'uso dell'
                                operatore. (Tabella C)
        - Sequence descriptor: f_=3, x_ e y_ indicano rispettivamente la classe del descrittore e
                               l'entry della classe come per Element descriptor (in questo caso però
                               si identificano gli elementi di Tabella D)
        Nel caso di dati prodotti da centri di emissioni locali che necessitano di una rappresentazione
        non definita nella Tabelle B, si riserva una parte delle tabelle B e D ad uso locale.

        Nota: f_ è rappresentato come numero a 2bits, x_ come numero a 6bits, y_ come numero a 8bits
        (********************************fine source documentazione***********************)

        In particolare in questa implementazione:
        f_ = primi 2 caratteri della stringa line in input.
        y_ = sottostringa dal 3° al 5° carattere di line.
        x_= sottostringa dal 6° al 9° carattere di line.
        value = sottostringa dal 10° al 23° carattere di line tranne quando (f_=='3',x_=='21',y_=='193')
        id5 = sottostringa dal 24° al 27° carattere di line tranne quando (f_=='3',x_=='21',y_=='193')
        id6 = sottostringa dal 28° al 32° carattere di line tranne quando (f_=='3',x_=='21',y_=='193')
        id7 = sottostringa dal 33° al 35° carattere di line tranne quando (f_=='3',x_=='21',y_=='193')
        descriptor = sottostringa dal 36° carattere di line tranne quando (f_=='3',x_=='21',y_=='193')

        infatti quando (f_=='3',x_=='21',y_=='193') la sottostringa successiva a 'f_ x_ y_' contiene il nome
        del file di metadati in input e settiamo le componenti id5,id6,id7 della lista octect a None, la
        componente value pari al resto della sottostringa e la componente descriptor='namefile'.
        """

        f_ = line[:col_ends[0]].strip()
        x_ = line[col_ends[0]:col_ends[1]].strip()  # 6bit
        y_ = line[col_ends[1]:col_ends[2]].strip()  # 8bit
        if f_ == "3" and x_ == "21" and y_ == "193":
            value = line[col_ends[2]:].strip()
            id5 = ""
            id6 = ""
            id7 = ""
            descriptor = "namefile"
        else:
            value = line[col_ends[2]:col_ends[3]].strip()
            id5 = line[col_ends[3]:col_ends[4]].strip()
            id6 = line[col_ends[4]:col_ends[5]].strip()
            id7 = line[col_ends[5]:col_ends[6]].strip()
            descriptor = line[col_ends[6]:].strip()

        octect = [f_, x_, y_, value, id5, id6, id7, descriptor]
        return [self._check_emptyfield(o) for o in octect]
    
    def read_meta(self, fname_meta: str) -> pd.DataFrame:

        """
        Metodo che legge il file di metadati decodificato dal BUFR tramite decbufr e
        restituisce un oggetto DataFrame di pandas, con le info sui metadati del BUFR.

        INPUT:
        - fname_meta    --str : nome del file di testo con i metadati del file BUFR
                                (ottenuto dalla decodifica tramite decbufr, es:
                                nomebufr.txt).

        OUTPUT:
        - df            --pd.DataFrame : oggetto DataFrame di pandas, avente colonne
                                         ["F", "X", "Y", "value", "id5", "id6", "id7", "descriptor"].
                                         Ciascuna riga del DataFrame ha 8 elementi ( uno per ogni
                                         colonna) i quali sono l'ottetto corrispondente ad una riga
                                         del file di metadati (nomebufr.dat ottenuto con decbufr).
                                         L'ottetto di descrittori su ciascuna riga del file dei metadati
                                         è letta tramite il metodo di istanza della classe Bufr
                                         simcradarlib.io_utils.bufr_class.Bufr.get_octect .
                                         Ciascun ottetto viene inserito come nuova riga del DataFrame
                                         'df'.

                                         In particolare per ogni riga, nella colonna "F" viene
                                         inserito il valore del descrittore f_, documentato in
                                         simcradarlib.io_utils.bufr_class.Bufr._check_emptyfield,
                                         di ciascun ottetto del file dei metadati.
                                         Analogamente la colonna "X" contiene i valori del descrittore
                                         x_ , la colonna "Y" i valori del descrittore y_, le
                                         colonna "id5" "id6 "id7" contengono rispettivamente i valori
                                         delle sottostringhe dal 24° al 27°, dal 28° al 32°,
                                         dal 33° al 35° carattere di ciascuna riga del file dei
                                         metadati. La colonna "descriptor" contiene i valori dalla
                                         36° carattere di ogni riga del file dei metadati.
                                         Per maggiori dettagli (e eccezioni) consultare la
                                         documentazione del metodo
                                         simcradarlib.io_utils.bufr_class.Bufr._check_emptyfield .
        """

        with open(fname_meta) as file:
            lines = [line for line in file]

        # infer last column index from the first line
        last_col_idx = lines[0].find("WMO block number")
        assert last_col_idx != -1, "The first line of the metadata file does not start with 'WMO block number'"

        # find the first line with 8 columns
        for line in lines:
            good_line = line[:last_col_idx]
            splitted = good_line.split()
            if len(splitted) == 7:
                break
        else:
            raise ValueError("No line with 8 columns found in the metadata file")

        # find the column width for each column including spaces
        cols_width = []
        for col in splitted:
            col_width = good_line.find(col) + len(col)
            cols_width.append(col_width)
            good_line = good_line[col_width:]

        col_ends = np.cumsum(cols_width)

        dfb = []
        for line in lines:
            dfb.append(self.get_octect(line, col_ends))
        df = pd.DataFrame(dfb, columns=["F", "X", "Y", "value", "id5", "id6", "id7", "descriptor"])

        return df

    def __del__(self):
        # Pulizia della directory temporanea se è stata creata
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()


def convert_bufr(
        in_path: str,
        out_path: Optional[str] = None,
        decbufr_path_or_obj: Optional[Union[str, Decbufr]] = None,
        tabdir: Optional[str] = None,
        fill_method: str = "ave",
        format: str = "geotiff",
        exit_on_error: bool = False,
        progress: bool = False,
    ) -> None:
    """
    Metodo per convertire uno o più BUFR nello standard Opera in GeoTiff o NetCDF.

    INPUT:
    - in_path           --str : percorso del file BUFR o della cartella contenente i BUFR da processare.
    - out_path          --str : percorso del file di output o della cartella in cui salvare i file convertiti.
                                Se non specificato, i file convertiti vengono salvati nella stessa cartella di in_path.
                                Se la cartella non esiste, viene creata.
    - decbufr_path_or_obj --Union[str, Decbufr] : percorso dell'eseguibile decbufr o oggetto Decbufr.
                                                  Se non specificato, viene utilizzato il file decbufr.tgz presente nella stessa directory del modulo.
    - tabdir            --str : percorso della directory contenente le tabelle BUFR.
                                Se non specificato, viene utilizzata la directory temporanea creata durante l'inizializzazione. 
    - fill_method       --str : metodo di riempimento dei livelli.
                                I valori possibili sono 'min', 'ave', 'max'.
                                Default: 'ave'.
    - format            --str : formato di output dei file convertiti.  
                                I valori possibili sono 'geotiff' e 'netcdf'. 'geotiff' richiede il modulo rasterio, 'netcdf' richiede il modulo netCDF4.
                                Default: 'geotiff'.
    - exit_on_error     --bool : se True, il programma termina in caso di errore durante la conversione di un file.
                                 Default: False.
    - progress          --bool : se True, mostra una barra di avanzamento durante la conversione. Richiede il modulo tqdm.
                                 Default: False.
    """ 

    assert format in ["geotiff", "netcdf"], "The 'format' argument must be either 'geotiff' or 'netcdf'."

    # Check if the input path is a file or a directory
    if os.path.isfile(in_path):
        bufr_files = [in_path]
    elif os.path.isdir(in_path):
        bufr_files = [os.path.join(in_path, f) for f in os.listdir(in_path) if f.endswith('.bufr') or f.endswith('.BUFR') or f.endswith('.Bufr')]
    else:
        raise FileNotFoundError(f"The input path '{in_path}' does not exist.")

    # Check if the output path is a file or a directory
    if out_path is not None:
        if os.path.isfile(out_path):
            raise FileExistsError(f"The output path '{out_path}' is an existing file.")
        elif not os.path.exists(out_path):
            os.makedirs(out_path)
    else:
        out_path = os.path.dirname(in_path)

    # Initialize the decbufr object
    delete_on_exit = False
    if decbufr_path_or_obj is None:
        decbufr = Decbufr()
        delete_on_exit = True
    elif isinstance(decbufr_path_or_obj, str):
        decbufr = Decbufr(decbufr_path_or_obj)
        delete_on_exit = True
    elif isinstance(decbufr_path_or_obj, Decbufr):
        decbufr = decbufr_path_or_obj
    else:
        raise ValueError("The decbufr_path_or_obj argument must be a string or an instance of Decbufr.")

    # Process each BUFR file
    if progress:
        import tqdm
        print(f"Processing {len(bufr_files)} BUFR(s)...")
        bufr_files = tqdm.tqdm(bufr_files, desc="Progress ", unit="file")
    for bufr_file in bufr_files:
        try:
            opera_bufr = decbufr.read_bufr(bufr_file, tabdir=tabdir, fill_method=fill_method)
            if format == "geotiff":
                out_file = os.path.join(out_path, os.path.basename(bufr_file)[0:-5] + '.tif')
                opera_bufr.to_geotiff(out_file)
            elif format == "netcdf":
                out_file = os.path.join(out_path, os.path.basename(bufr_file)[0:-5] + '.nc')
                opera_bufr.to_netcdf(out_file)
        except Exception as e:
            if exit_on_error:
                raise e
            else:
                print(f"Error processing '{bufr_file}': {e}")

    # Cleanup the decbufr object if it was created in this function
    if delete_on_exit:
        del decbufr


def main():
    # Import the Fire library
    import fire
    
    # we use the Fire library to automatically generate a CLI for the convert_bufr function
    fire.Fire(convert_bufr)

if __name__ == "__main__":
    main()
