USEDTREATMENTS = 2
USEDWEEKS = 4
TREATMENT_TYPES = ["PGE2", "HPI4"]
WS = ["W1", "W2", "W3", "W4", "W5", "W6"]

XSCALE, YSCALE, ZSCALE = 0.21666666666666667, 0.21666666666666667, 0.500
VOLUMESCALE = XSCALE * YSCALE * ZSCALE
AREASCALE = XSCALE * YSCALE


def generate_repinfo(_alphabets=["B", "C", "D", "E", "F", "G"]):
    """
    Input one of the alphabets based on specific file information
    :param _alphabets:
    :return:
    """
    _repnums = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    reps = []
    for a in _alphabets:
        for repnum in _repnums:
            reps.append(a + repnum)
    return reps


def getwr(df, af, lf):
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


def findtreatment(r):  # TODO: check with getwr for inconsistencies
    treatment = None
    if r <= 5:
        treatment = 0
    else:
        treatment = 1
    return treatment


def checkcellconditions(cellvals):
    [centroid, vol, xspan, yspan, zspan, maxferet, minferet] = cellvals
    satisfied_conditions = True
    if (zspan < 1) or (xspan < 2) or (yspan < 2):
        satisfied_conditions = False
    if vol >= 100000 or vol <= 50:  # 50 = 2130,in cu micron
        satisfied_conditions = False
    return satisfied_conditions


# TODO: ?
class channels():
    def __init__(self):
        pass

    def getchannelnames(self):
        channelnames = {
            "TOM20": "TOM20"
        }
