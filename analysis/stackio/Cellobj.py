class Cell:
    def __init__(self, inputchannelname=None):
        self.gfp = GFP()
        self.selected = True  # by default
        self.nuclei = Nucleus()
        # generate cell, dna and gfp
        pass

    @property
    def id(self):
        pass

    @property
    def volume(self):
        pass

    @property
    def xspan(self):
        pass

    @property
    def yspan(self):
        pass

    @property
    def zspan(self):
        pass

    @property
    def minferet(self):
        pass

    @property
    def maxferet(self):
        pass

    @property
    def aspectratio(self):
        pass

    @property
    def MIParea(self):
        pass

    @property
    def centroid(self):
        pass

    @property
    def selected(self):
        return self.selected


class Nucleus():
    def __init__(self):
        # generate cell, dna and gfp
        pass


class GFP():
    def __init__(self, inputchannelname=None):
        # generate cell, dna and gfp
        pass
