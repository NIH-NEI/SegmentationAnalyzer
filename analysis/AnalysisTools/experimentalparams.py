from .dtypes import *

USEDTREATMENTS = 2
USEDWEEKS = 4  # weeks used for calculations: 1-4
USEDWELLS = 5
WELLS = ["well_1", "well_2", "well_3", "well_4", "well_5"]  # placeholder names. Use generate repinfo whenever necessary
TREATMENT_TYPES = ["PGE2", "HPI4"]
WS = ["W1", "W2", "W3", "W4", "W5", "W6"]  # Weeks
FIELDSOFVIEW = ["F001", "F002", "F003", "F004", "F005", "F006"]  # Fields of view
TOTALFIELDSOFVIEW = len(FIELDSOFVIEW)
XSCALE, YSCALE, ZSCALE = 0.21666666666666667, 0.21666666666666667, 0.5
VOLUMESCALE = XSCALE * YSCALE * ZSCALE
AREASCALE = XSCALE * YSCALE

MAX_CELLS_PER_STACK = 1000
MAX_DNA_PER_CELL = 2
MAX_ORGANELLE_PER_CELL = 250  # TENTATIVE on channel
"""
MAX_ORGANELLE_PER_CELL
used:
100 for TOM
200 for Lamp1
250 for Sec
"""

TIMEPOINTS_PER_STACK = 1
CHANNELLS_PER_STACK = 4
Z_FRAMES_PER_STACK = 27
Y_PIXELS_PER_STACK = 1078
X_PIXELS_PER_STACK = 1278

T = TIMEPOINTS_PER_STACK
C = CHANNELLS_PER_STACK
Z = Z_FRAMES_PER_STACK
Y = Y_PIXELS_PER_STACK
X = X_PIXELS_PER_STACK

STACK_DIMENSION_ORDER = "(T, C, Z, X, Y)"
ORIGINAL_STACK_SHAPE = (T, C, Z, X, Y)


def getunits(propertyname):
    units = {
        "Centroid": "microns",
        "Volume": "cu. microns",
        "Mean Volume": "cu. microns",
        "X span": "microns",
        "Y span": "microns",
        "Z span": "microns",
        "MIP area": "sq. microns",
        "Max feret": "microns",
        "Min feret": "microns",
        "2D Aspect ratio": None,
        "Volume fraction": "percent",
        "Count per cell": None,
        "Orientation": None,
        "z-distribution": "microns",
        "radial distribution 2D": "microns",
        "normalized radial distribution 2D": None,
        "radial distribution 3D": "microns",
        "Sphericity": None
    }
    print("PROPERTYNAME:", propertyname, "UNITS:", propertyname in units.keys())
    return units[propertyname]


def replicate_info(_alphabets: strList = None) -> list:
    """
    Input one of the alphabets based on specific file information
    
    Args:
        _alphabets:
    Returns:
         list of replicate ids
    """
    allalphabets = ["B", "C", "D", "E", "F", "G"]
    if _alphabets is None:
        _alphabets = allalphabets
    else:
        assert _alphabets in allalphabets, f"Alphabets must be from {allalphabets}"
    _repnums = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    reps = []
    for a in _alphabets:
        for repnum in _repnums:
            reps.append(a + repnum)
    return reps


def find_treatment(r):
    """
    Returns the type of treatement based on replicate id
    Args:
        r: replicate id ( converted to 0-9 range)
    Returns:
         treatment id
    """
    assert r < 10, "r must be in range 0-9"
    treatment = r // 5
    return treatment


def find_week(filename: PathLike) -> str:
    """
    Returns week number

    Args:
        filename: filename

    Returns:
        week number from filename
    """
    all_weeks = []
    for w in WS:
        if w in filename:
            all_weeks.append(w)
    if len(all_weeks) != 1:
        print("all_weeks : ", all_weeks)
        raise Exception
    return all_weeks[0]


def findrep(filename: str, _alphabets=None):
    """
    Use given alphabet for finding replicate number

    Args:
        filename: filename
        _alphabets: this is based on nomenclature used in naming files

    Returns:
        replicate number
    """
    all_reps = []
    if _alphabets is None:
        _alphabets = ["B", "C", "D", "E", "F", "G"]
        reps = replicate_info(_alphabets=_alphabets)
    else:
        raise Exception
    for r in reps:
        if r in filename:
            all_reps.append(r)
    if len(all_reps) != 1:
        print("all_reps : ", all_reps)
        raise Exception
    else:
        return int(all_reps[0][1:]) - 2


def getusedchannels(filelist: str) -> list:
    """
    Get list of channels used in given list of files

    Args:
    filelist: list of filenames

    Returns:
         list of channels
    """
    channels = []
    for file in filelist:
        channel = file.split("_")[0].split("-")[-1]
        if channel not in channels:
            channels.append(channel)
    return channels


def check_sufficient_datapoints_per_stack(vols: list, x_spans: list, y_spans: list, z_spans: list,
                                          mip_areas: list) -> bool:
    """
    Only choose stacks with 10 or more datapoints in them - this will help reduce bias from stacks with too few cells

    Args:
        vols: list of corresponding volumes
        x_spans: list of corresponding x spans
        y_spans:list of corresponding y spans
        z_spans: list of corresponding z spans
        mip_areas: list of corresponding mip areas

    Returns:
         bool: are datapoints sufficient for stack based data?
    """
    sufficient_datapoints = True
    assert (len(vols) == len(x_spans) == len(y_spans) == len(z_spans) == len(mip_areas))
    if len(vols) <= 10:
        sufficient_datapoints = False
    return sufficient_datapoints


def cell_biologically_valid(cell_vals, remove_cut_cells=True, vol_cutoff=50, debug=False) -> (bool, bool):
    """
        Checks if cell (Actin/outer border enclosed object) meets minimum requirements chosen based on
    expert knowledge. This can be used to filter bad segmentations of cell data.

    Args:
        cell_vals: list of values -> centroid, vol, x_span, y_span, z_span, max_feret, min_feret, mip_area, cell touching top(bool),cell touching bot(bool).
        remove_cut_cells: True by default. Any cells touching top or bot are removed.
        vol_cutoff: cutoff volume in cu. microns
        debug: print out values for cell properties

    Returns:
         True if all conditions are met. False if any is not met (indicating biologically impossible segmentation.)
    """
    [centroid, vol, x_span, y_span, z_span, max_feret, min_feret, mip_area, top, bot] = cell_vals
    if debug:
        print(centroid, vol, x_span, y_span, z_span, max_feret, min_feret, mip_area, top, bot)
    satisfied_conditions = True
    cut = False
    if (top or bot) and remove_cut_cells:
        satisfied_conditions = False
        cut = True
    if (z_span <= 1.0) or (x_span <= 1.5) or (y_span <= 1.5) or (min_feret <= 1.5):
        satisfied_conditions = False
    if vol >= 100000 or vol <= vol_cutoff:  # 50 = 2130, 10 = 426
        satisfied_conditions = False
    # print("CHECK:", satisfied_conditions, cut, top, bot)
    return satisfied_conditions, cut


class Channel:
    def __init__(self, input_channel_name=None):
        self.all_channel_names = ["dna", "actin", "membrane", "tom20", "pxn", "sec61b", "tuba1b",
                                  "lmnb1", "fbl", "actb", "dsp", "lamp1", "tjp1", "myh10", "st6gal1",
                                  "lc3b", "cetn2", "slc25a17", "rab5", "gja1", "ctnnb1"]
        self.organelle_structure = {
            "dna": "Nucleus",  # check
            "actin": "Actin Filaments",
            "membrane": "Cell membrane",  # check
            "tom20": "Mitochondria",
            "pxn": "Matrix adhesions",
            "sec61b": "Endoplasmic reticulum",
            "tuba1b": "Microtubules",
            "lmnb1": "Nuclear Envelope",
            "fbl": "Nucleolus",
            "actb": "Actin Filaments",
            "dsp": "Desmosomes",
            "lamp1": "Lysosome",
            "tjp1": "Tight Junctions",
            "myh10": "Actomyosin bundles",
            "st6gal1": "Golgi Apparatus",
            "lc3b": "Autophagosomes",
            "cetn2": "Centrioles",
            "slc25a17": "Peroxisomes",
            "rab5": "Endosomes",
            "gja1": "Gap Junctions",
            "ctnnb1": "Adherens Junctions"
        }
        self.rep_alphabet = {
            "dna": "default",
            "actin": "default",
            "membrane": "default",
            "tom20": "E",
            "pxn": "F",
            "sec61b": "G",
            "tuba1b": "C",
            "lmnb1": "F",
            "fbl": "G",
            "actb": "D",
            "dsp": "B",
            "lamp1": "B",
            "tjp1": "D",
            "myh10": "F",
            "st6gal1": "C",
            "lc3b": "C",
            "cetn2": "G",
            "slc25a17": "E",
            "rab5": "E",
            "gja1": "G",
            "ctnnb1": "F"
        }
        self.channel_protein = {
            "dna": "",
            "actin": "Beta-actin",
            "membrane": "",
            "tom20": "tom20",
            "pxn": "Paxillin",
            "sec61b": "",
            "tuba1b": "Alpha Tubulin",
            "lmnb1": "Lamin B1",
            "fbl": "Fibrillarin",
            "actb": "Beta-actin",
            "dsp": "Desmoplakin",
            "lamp1": "LAMP-1",
            "tjp1": "Tight Junction Protein Z-01",
            "myh10": "Non-muscle myosin heavy chain IIB",
            "st6gal1": "Sialyltransferase 1",
            "lc3b": "Autophagy-related protein LC3 B",
            "cetn2": "Centrin-2",
            "slc25a17": "Peroxisomal membrane protein",
            "rab5": "Ras-related protein Rab-5A",
            "gja1": "Connxin-43",
            "ctnnb1": "Beta-catenin"
        }
        if self.validchannelname(input_channel_name):
            self.channel_name = input_channel_name
            self.channel_protein = self.getproteinname(input_channel_name)
            self.organelle_structure_name = self.getorganellestructurename(input_channel_name)
            self.rep_alphabet = self.getrepalphabet(input_channel_name)
        else:
            raise Exception(
                f"Invalid Channel name:{input_channel_name}. Name must be one of {self.all_channel_names}")
        self.minarea = {
            "dna": 4,
            "actin": 4,
            "membrane": 4,
            "tom20": 4,
            "pxn": 4,
            "sec61b": 4,
            "tuba1b": 4,
            "lmnb1": 4,
            "fbl": 4,
            "actb": 0,
            "dsp": 0,
            "lamp1": 4,
            "tjp1": 4,
            "myh10": 0,
            "st6gal1": 4,
            "lc3b": 4,
            "cetn2": 4,
            "slc25a17": 0,
            "rab5": 4,
            "gja1": 0,
            "ctnnb1": 0
        }
        self.directory = None

    def getallallchannelnames(self):
        """
        Returns
            List of all channel names
        """
        return self.all_channel_names

    def getminarea(self, key):
        """
        returns minarea for channel

        Args:
            key: channelname
        Returns:
            minarea for requested channel
        """
        return self.minarea[key]

    def getproteinname(self, key):
        """
        returns protein name for channel

        Args:
            key: channelname
        Returns:
            protein name for requested channel
        """
        return self.channel_protein[key]

    def getorganellestructurename(self, key):
        """
        returns organelle structure name for channel

        Args:
            key: channelname
        Returns:
            organelle structure name for requested channel
        """
        return self.organelle_structure[key]

    def validchannelname(self, key):
        """
        returns if channel name is valid (known to class)

        Args:
            key: channelname
        Returns:
            does channel exist in class
        """
        return key.lower() in self.all_channel_names

    def getrepalphabet(self, key):
        """
        returns naming convention alphabet for channel

        Args:
            key: channelname
        Returns:
             naming convention alphabet for requested channel
        """
        return self.rep_alphabet[key]

    def setdirectoryname(self, channel, dname):
        """
        Use to set directory name in case it is different from channelname
        channel:
        Args:
            channel: channel
            dname: directory name
        Returns:
            None
        """
        self.directory = {
            "lmnb1": "LaminB",
            "lamp1": "LAMP1",
            "sec61b": "Sec61",
            "st6gal1": "ST6GAL1",
            "tom20": "TOM20",
            "fbl": "FBL",
            "myh10": "myosin",
            "rab5": "RAB5",
            "tuba1b": "TUBA",
            "dsp": "DSP",
            "slc25a17": "SLC",
            "pxn": "PXN",
            "gja1": "GJA1",
            "ctnnb1": "CTNNB",
            "actb": "ACTB",
            "cetn2": "CETN2",
            "lc3b": "LC3B"}

        assert isinstance(dname, str)
        self.directory[channel] = dname
