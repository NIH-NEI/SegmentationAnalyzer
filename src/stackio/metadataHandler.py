import json
from src.AnalysisTools import experimentalparams as ep


# def createstackjson(path, ):
#     with open(path) as jsonfile:
#         json.dump()

def createcelldict(id, parent=None, xspan=None, yspan=None, zspan=None, centroid=None, volume=None,
                   mip_area=None, edgetag=None):
    """
    Creates and returns a dictinary object by assigning and Id to each cell and associating its
    properties with the id. This metadata can be saved for future use.

    :param id: cell id within stack
    :param parent: stack id
    :param xspan: feret measurement along x axis
    :param yspan: feret measurement along y axis
    :param zspan: feret measurement along z axis
    :param centroid: centroid (z,y,x)
    :param volume: cell volume
    :param mip_area:
    :param edgetag:
    :return:
    """
    if (id is None) or (parent is None) or (xspan is None) or (yspan is None) or (
            zspan is None) or (centroid is None) or (volume is None) or (mip_area is None) or (
            edgetag is None):
        # raise incompleteinputException
        print(
            f"id = {id}, parent={parent}, xspan={xspan}, yspan={yspan}, zspan={zspan}, centroid={centroid}, volume={volume}, mip_area={mip_area}, edgetag={edgetag}")
        raise Exception

    celldict = {'id': id,
                'parent': parent,
                'xspan': int(xspan),
                'yspan': int(yspan),
                'zspan': int(zspan),
                'volume': int(volume),
                'edgetag': edgetag,
                'mip_area': int(mip_area),
                'centroid': list(centroid)}
    return celldict

class cellobject():
    def __init__(self, inputchannelname=None):
        if self.validchannelname(inputchannelname):
            channelinfo = ep.Channel(inputchannelname)
            self.channelname = inputchannelname
            self.channelprotein = channelinfo.getproteinname(inputchannelname)
            self.organellestructurename = channelinfo.getorganellestructurename(inputchannelname)
            self.repalphabet = channelinfo.getrepalphabet(inputchannelname)

def writeCellMetadata(jsonfile, individualcelldict):
    try:
        with open(jsonfile, 'w') as jf:
            json.dump(individualcelldict, jf, indent=1)
        # id = None, stackname = None, volume = None, xspan = None, yspan = None, zspan = None, centroid = None
        print(jsonfile, "dumped")
        return 1
    except Exception as e:
        print(e)
        return 0


def readCellMetadata(jsonfile):
    try:
        with open(jsonfile) as jf:
            a = json.load(jf)
            return a
    except:
        return 0


def parsejson(info):
    vol = info['volume'] * ep.VOLUMESCALE
    xspan = info['xspan'] * ep.XSCALE
    yspan = info['yspan'] * ep.YSCALE
    zspan = info['zspan'] * ep.ZSCALE
    miparea = info['mip_area'] * ep.AREASCALE
    tag = info['edgetag']
    top, bot = 0, 0
    #     print(tag,end =" ")
    if tag.__contains__('b'):
        bot = 1
    if tag.__contains__('t'):
        top = 1
    if tag.__contains__('n'):
        top, bot = 0, 0
    return vol, xspan, yspan, zspan, miparea, top, bot


if __name__ == "__main__":
    wpath = "../Results/2021/mar24/jsontest/"
    jsonfile = wpath + "test.json"
    allcells = []
    for i in range(10):
        Cdict = createcelldict(1, f's{i}', 2, 3, 4, [5, 5, 5], 66, 51)
        # print(Cdict)
        allcells.append(Cdict)

    writeCellMetadata(jsonfile, allcells)
    ans = readCellMetadata(jsonfile)
    print(type(ans), len(ans), allcells, "\n", ans)
    print("Read == Write ?: ", allcells == ans)
