from src.AnalysisTools import types

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
MAX_ORGANELLE_PER_CELL = 100  # TENTATIVE on channel

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


def generate_repinfo(_alphabets: types.strlist = None):
    """
    Input one of the alphabets based on specific file information

    :param _alphabets:
    :return: list of replicate ids
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


def findtreatment(r):  # TODO: check with getwr_3channel for inconsistencies
    """
    Returns the type of treatement based on replicate id
    :param r: replicate id ( converted to 0-9 range)
    :return: treatment id
    """
    assert r < 10, "r must be in range 0-9"
    treatment = r // 5
    return treatment


def findweek(filename):
    all_weeks = []
    for w in WS:
        if w in filename:
            all_weeks.append(w)
    if len(all_weeks) != 1:
        print("all_weeks : ", all_weeks)
        raise Exception
    return all_weeks[0]


def findrep(filename, _alphabets=None):
    """
    Use given alphabet for
    :param filename:
    :param _alphabets:
    :return:
    """
    all_reps = []
    if _alphabets is None:
        _alphabets = ["B", "C", "D", "E", "F", "G"]
        reps = generate_repinfo(_alphabets=_alphabets)
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


#     return all_reps[0]

def getusedchannels(filelist):
    channels = []
    for file in filelist:
        channel = file.split("_")[0].split("-")[-1]
        if channel not in channels:
            channels.append(channel)
    return channels


def checkstackconditions(vols, xspans, yspans, zspans, mipareas):
    """
    TODO: finalize
    Only choose stacks with 10 or more datapoints in them - this will help reduce bias from stacks with too few cells
    :param vols:
    :param xspans:
    :param yspans:
    :param zspans:
    :param mipareas:
    :return:
    """
    stackconditions = True
    assert (len(vols) == len(xspans) == len(yspans) == len(zspans) == len(mipareas))
    if len(vols) <= 10:
        stackconditions = False
    return stackconditions


def checkcellconditions(cellvals, removecutcells=True, volcutoff=50):
    """
        Checks if cell (Actin/outer border enclosed object) meets minimum requirements chosen based on
    expert knowledge. This can be used to filter bad segmentations of cell data.

    :param cellvals: list of values -> centroid, vol, xspan, yspan, zspan, maxferet, minferet, miparea, cell touching top(bool),cell touching bot(bool).
    :param removecutcells: True by default. Any cells touching top or bot are removed.
    :param volcutoff: cutoff volume in cu. microns
    :return: True if all conditions are met. False if any is not met (indicating biologically
    impossible segmentation.)
    """
    [centroid, vol, xspan, yspan, zspan, maxferet, minferet, miparea, top, bot] = cellvals
    # print(centroid, vol, xspan, yspan, zspan, maxferet, minferet, miparea, top, bot)
    satisfied_conditions = True
    cut = 0
    if (top or bot) and removecutcells:
        satisfied_conditions = False
        cut = 1
    if (zspan <= 1.0) or (xspan <= 1.5) or (yspan <= 1.5) or (minferet <= 1.5):
        satisfied_conditions = False
    if vol >= 100000 or vol <= volcutoff:  # 50 = 2130, 10 = 426
        satisfied_conditions = False
    # print("CHECK:", satisfied_conditions, cut, top, bot)
    return satisfied_conditions, cut


class channel():
    def __init__(self, inputchannelname=None):
        # TODO: refactor to static
        self.allchannelnames = ["dna", "actin", "membrane", "tom20", "pxn", "sec61b", "tuba1b",
                                "lmnb1", "fbl", "actb", "dsp", "lamp1", "tjp1", "myh10", "st6gal1",
                                "lc3b", "cetn2", "slc25a17", "rab5", "gja1", "ctnnb1"]
        self.organellestructure = {
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
        self.channelprotein = {
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
        if self.validchannelname(inputchannelname):
            self.channelname = inputchannelname
            self.channelprotein = self.getproteinname(inputchannelname)
            self.organellestructurename = self.getorganellestructurename(inputchannelname)
            self.repalphabet = self.getrepalphabet(inputchannelname)
        else:
            raise Exception(
                f"Invalid Channel name:{inputchannelname}. Name must be one of {self.allchannelnames}")
        self.minarea = {  # TODO: get values from current segmenter or remove
            "dna": 4,
            "actin": 4,
            "membrane": 4,
            "tom20": 4,
            "pxn": 4,
            "sec61b": 4,
            "tuba1b": 4,
            "lmnb1": 4,
            "fbl": 4,
            "actb": 4,
            "dsp": 4,
            "lamp1": 4,
            "tjp1": 4,
            "myh10": 4,
            "st6gal1": 4,
            "lc3b": 4,
            "cetn2": 4,
            "slc25a17": 4,
            "rab5": 4,
            "gja1": 4,
            "ctnnb1": 4
        }
        self.directory = None

    def getallallchannelnames(self):
        return self.allchannelnames

    def getminarea(self, key):
        return self.minarea[key]

    def getproteinname(self, key):
        return self.channelprotein[key]

    def getorganellestructurename(self, key):
        return self.organellestructure[key]

    def validchannelname(self, key):
        return key.lower() in self.allchannelnames

    def getrepalphabet(self, key):
        return self.rep_alphabet[key]

    def setdirectoryname(self, channel, dname):
        """
        Use to set directory name in case it is different from channelname
        :param channel:
        :param dname:
        :return:
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
