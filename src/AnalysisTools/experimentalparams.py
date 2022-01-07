from src.AnalysisTools import types

USEDTREATMENTS = 2
USEDWEEKS = 4
TREATMENT_TYPES = ["PGE2", "HPI4"]
WS = ["W1", "W2", "W3", "W4", "W5", "W6"]
FIDS = ["F001","F002","F003","F004","F005","F006"]

XSCALE, YSCALE, ZSCALE = 0.21666666666666667, 0.21666666666666667, 0.500
VOLUMESCALE = XSCALE * YSCALE * ZSCALE
AREASCALE = XSCALE * YSCALE
T = 1
C = 4
Z = 27
Y = 1078
X = 1278
setstackshape = (T, C, Z, X, Y)


def generate_repinfo(_alphabets: types.strlist = ["B", "C", "D", "E", "F", "G"]):
    """
    Input one of the alphabets based on specific file information

    :param _alphabets:
    :return: list of replicate ids
    """
    allalphabets = ["B", "C", "D", "E", "F", "G"]
    if _alphabets is None:
        _alphabets = allalphabets
    else:
        assert (_alphabets in allalphabets, f"Alphabets must be from {allalphabets}")
    _repnums = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    reps = []
    for a in _alphabets:
        for repnum in _repnums:
            reps.append(a + repnum)
    return reps


def getwr_3channel(df, af, lf):  # TODO:test
    """
    Checks names of files and ensures the files correspond to each other. Returns week and replicate
    information for calculation purposes.

    :param df: dnafilenames
    :param af: actinfilenames
    :param lf: gfp_channel_filenames
    :return: week id, replicate id, week no. replicate number, common base string
    """
    basestringdna = "_".join(df.split("_")[:-2])
    basestringactin = "_".join(af.split("_")[:-3])
    basesstringlmp = "_".join(lf.split("_")[:-1])

    print(basestringdna, basestringactin, basesstringlmp)
    assert basestringdna == basestringactin == basesstringlmp
    s1, r, _ = basestringdna.split("_")
    w = s1.split("-")[1]
    w_ = WS.index(w)
    r_ = int(r[1:]) - 2
    return w, r, w_, r_, basestringdna


def findtreatment(r):  # TODO: check with getwr_3channel for inconsistencies
    """
    Returns the type of treatement based on replicate id
    :param r: replicate id ( converted to 0-9 range)
    :return: treatment id
    """
    assert (r < 10, "error")
    treatment = None
    if r < 5:
        treatment = 0
    else:
        treatment = 1
    return treatment


def checkcellconditions(cellvals):
    """
    Checks if cell (Actin/outer border enclosed object) meets minimum requirements chosen based on
    expert knowledge. This can be used to filter bad segmentations of cell data.

    :param cellvals: list of values -> centroid, volume, xspan, yspan, zspan, maximum feret, minimum feret.
    :return: True if all conditions are met. False if any is not met (indicating biologically
    impossible segmentation.)
    """
    [centroid, vol, xspan, yspan, zspan, maxferet, minferet] = cellvals
    satisfied_conditions = True
    if (zspan < 1) or (xspan < 2) or (yspan < 2):
        satisfied_conditions = False
    if vol >= 100000 or vol <= 50:  # 50 = 2130,in cu micron
        satisfied_conditions = False
    return satisfied_conditions


# TODO: ?


class channel():
    def __init__(self, key=None):
        self.allchannelnames = ["DNA", "Actin", "Membrane", "TOM20", "PXN", "SEC61B", "TOM20", "TUBA1B", "LMNB1", "FBL", "ACTB", "DSP", "LAMP1", "TJP1", "MYH10", "ST6GAL1", "LC3B", "CETN2",
                                "SLC25A17",
                                "RAB5", "GJA1", "CTNNB1"]

        self.organellestructure = {
            "DNA": "Nucleus",  # check
            "Actin": "Actin Filaments",
            "Membrane": "Cell membrane",  # check
            "TOM20": "Mitochondria",
            "PXN": "Matrix adhesions",
            "SEC61B": "Endoplasmic reticulum",
            "TUBA1B": "Microtubules",
            "LMNB1": "Nuclear Envelope",
            "FBL": "Nucleolus",
            "ACTB": "Actin Filaments",
            "DSP": "Desmosomes",
            "LAMP1": "Lysosome",
            "TJP1": "Tight Junctions",
            "MYH10": "Actomyosin bundles",
            "ST6GAL1": "Golgi Apparatus",
            "LC3B": "Autophagosomes",
            "CETN2": "Centrioles",
            "SLC25A17": "Peroxisomes",
            "RAB5": "Endosomes",
            "GJA1": "Gap Junctions",
            "CTNNB1": "Adherens Junctions"
        }

        self.channelprotein = {
            "DNA": "",
            "Actin": "Beta-actin",
            "Membrane": "",
            "TOM20": "Tom20",
            "PXN": "Paxillin",
            "SEC61B": "",
            "TUBA1B": "Alpha Tubulin",
            "LMNB1": "Lamin B1",
            "FBL": "Fibrillarin",
            "ACTB": "Beta-actin",
            "DSP": "Desmoplakin",
            "LAMP1": "LAMP-1",
            "TJP1": "Tight Junction Protein Z-01",
            "MYH10": "Non-muscle myosin heavy chain IIB",
            "ST6GAL1": "Sialyltransferase 1",
            "LC3B": "Autophagy-related protein LC3 B",
            "CETN2": "Centrin-2",
            "SLC25A17": "Peroxisomal membrane protein",
            "RAB5": "Ras-related protein Rab-5A",
            "GJA1": "Connxin-43",
            "CTNNB1": "Beta-catenin"
        }
        if self.validchannelname(key):
            self.channelname = key
            self.channelprotein = self.getproteinname(key)
            self.organellestructurename = self.getorganellestructurename(key)
        else:
            raise Exception(f"Invalid Channel name:{key}. Name must be one of {self.allchannelnames}")


    def getallallchannelnames(self):
        return self.allchannelnames

    def getproteinname(self, key):
        return self.channelprotein[key]

    def getorganellestructurename(self, key):
        return self.organellestructure[key]

    def validchannelname(self, key):
        return key in self.allchannelnames
