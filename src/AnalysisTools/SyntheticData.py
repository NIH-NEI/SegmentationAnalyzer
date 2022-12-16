import math

import cv2
import numpy as np
from aicsimageio.writers import OmeTiffWriter
from skimage import draw


class SyntheticCell():
    def __init__(self):
        # generates a synthesized cell with 3 channels
        # - polygonal outside, elliptical inside and a distribution of a few particles

        pass

    def usedefaultparams(self):
        # sets all parameters to default
        pass

    @staticmethod
    def generate_synthetic_cell():
        ch1 = SyntheticCell.generatepolygonalcell()
        ch2 = SyntheticCell.generateellipticalshape()
        # ch3 = SyntheticCell.generatedistributedparticles()
        ch2_1 = np.zeros_like(ch1)
        ch2s = ch2.shape
        print(ch1.shape, ch2.shape, ch2_1.shape)  # , ch3.shape)
        # print(ch2_1[:,  :,4:ch2s[2] + 4, 30:ch2s[3] + 30, 30:ch2s[4] + 30].shape)
        ch2_1[:, :, 4:ch2s[2] + 4, 30:ch2s[3] + 30, 30:ch2s[4] + 30] = ch2
        ch2_1 = ch2_1 & ch1
        cell = np.concatenate((ch1, ch2_1), axis=1)
        print(cell.shape)
        return cell

    @staticmethod
    def standard_synthetic_cell(savename):
        stdcell = SyntheticCell.generate_synthetic_cell()
        OmeTiffWriter.save(data=stdcell, uri=savename, overwrite_file=True)
        return stdcell

    @staticmethod
    def generatepolygonalcell(size=(100, 100, 3), n=6, z=27, color=(255, 255, 255), center_coordinates=(50, 50),
                              rxy=40, minz=4, maxz=20, angle=0, filenum=0):
        """
        TODO: Optional generate a scutoid

        :param size: Size of the image
        :param n:
        :param z:
        :param color:
        :param center_coordinates:
        :param rxy:
        :param minz:
        :param maxz:
        :param angle:
        :param filenum:
        :return:
        """
        x, y = center_coordinates[0], center_coordinates[1]
        points = []
        irr = np.random.uniform(low=-90 / n, high=90 / n, size=n).tolist()
        irr = np.asarray(irr) * 2 * math.pi / 360
        #     print(irr)
        for i in range(n):
            point = [x + rxy * math.cos(2 * math.pi * i / n + (angle + irr[i])),
                     y + rxy * math.sin(2 * math.pi * i / n + (angle + irr[i]))]
            points.append(point)
        polypoints = np.asarray(points, np.int32)
        polypoints = polypoints.reshape((-1, 1, 2))
        ims = []

        for i in range(z):
            img = np.zeros(size, np.uint8)
            if minz < i <= maxz:
                img = cv2.fillPoly(img, [polypoints], color=(0, 255, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ims.append(img)
        images = np.asarray(ims)
        del ims
        expanded = np.expand_dims(images, 0)
        expanded = np.expand_dims(expanded, 0)
        expanded = expanded.transpose(0, 1, 2, 3, 4)
        return expanded
        # filename = f"poly{n}_{size[0]}_{z}_{x}_{y}_{rxy}_{minz}_{maxz}_{angle:0.2f}_{filenum}.tif"
        # OmeTiffWriter.save(data=expanded.transpose(0, 2, 1, 3, 4), uri=join(savepath, filename), compress=6)

    @staticmethod
    def generateellipticalshape(semimajorx=8, semimajory=15, semimajorz=15, offset=None):
        """
        Use for nucleus or elliptical particles
        :param semimajorx:
        :param semimajory:
        :param semimajorz:
        :param offset:
        :return:
        """
        # x, y = center_coordinates[0], center_coordinates[1]
        # points = []
        # for i in range(n):
        #     point = []
        if offset is None:  # ideally based on center of synthetic cell
            offset = [0, 0, 0]
        ellipse = draw.ellipsoid(semimajorx, semimajory, semimajorz) * 255
        # ellipse = ellipse
        print(ellipse.shape, np.unique(ellipse))
        # (N, M, P)
        expanded = np.expand_dims(ellipse, 0)
        expanded = np.expand_dims(expanded, 0)
        expanded = expanded.transpose(0, 1, 2, 3, 4)
        return expanded

        # @staticmethod
        # def generatedistributedparticles(userandom = False, cell):
        #     if userandom:
        #         mu, sigma = 0, 0.1
        #         nparticles = 20
        #         random.random.normal(mu, sigma , nparticles)
        #         # TODO: add steps to generate ellipses
        #     else:
        #         # generate 3 ellipses
        #         a = SyntheticCell.generateellipticalshape(semimajorx=2, semimajory=2, semimajorz=2, offset=[5,5,5])
        #         b = SyntheticCell.generateellipticalshape(semimajorx=2, semimajory=2, semimajorz=2, offset=[5,10,1])
        #         c = SyntheticCell.generateellipticalshape(semimajorx=2, semimajory=2, semimajorz=2, offset=[5,2,12])
        #         #assume standard
        #         cell
        #         # 3 z values and 3 xy values along a circle

        pass

    @staticmethod
    def generateparticle(self, objshape='elliptical'):

        pass

    @staticmethod
    def generatesimplecuboid(length: int = 20, breadth: int = 20, width: int = 20, refarray=None, loc='center'):
        if refarray is not None:
            maxdims = refarray.shape
        else:
            maxdims = (50, 50, 50)
        cube = np.zeros(maxdims)
        # print(maxdims, (maxdims[0] - 20) // 2, type(maxdims[0]), type(length))
        print(f"generating cuboid of sides: {length}, {breadth}, {width}")
        if loc == 'center':
            l1 = (maxdims[0] - length) // 2
            l2 = (maxdims[1] - breadth) // 2
            l3 = (maxdims[2] - width) // 2
            cube[l1:-l1, l2:-l2, l3:-l3] = 255
        else:
            assert isinstance(loc, tuple)
            c1 = maxdims[0] // 2 + loc[0]
            c2 = maxdims[1] // 2 + loc[1]
            c3 = maxdims[2] // 2 + loc[2]
            cube[c1:c1 + length, c2:c2 + breadth, c3:c3 + width] = 255
        return cube

    @staticmethod
    def generatecuboidwithparticle():
        cuboid = SyntheticCell.generatesimplecuboid()
        expandedcuboid = np.expand_dims(cuboid, 0)
        expandedcuboid = np.expand_dims(expandedcuboid, 0)
        expandedcuboid = expandedcuboid.transpose(0, 1, 2, 3, 4)

        # particle = SyntheticCell.generatesimplecuboid(length=2, breadth=2, width=2, refarray=cuboid, loc='center')
        particle = SyntheticCell.generatesimplecuboid(length=1, breadth=1, width=1, refarray=cuboid, loc=(4, 6, 8))
        expandedparticle = np.expand_dims(particle, 0)
        expandedparticle = np.expand_dims(expandedparticle, 0)
        expandedparticle = expandedparticle.transpose(0, 1, 2, 3, 4)
        cell = np.concatenate((expandedcuboid, expandedparticle), axis=1)
        return cell


if __name__ == "__main__":
    # savepath = "C:/Users/satheps/PycharmProjects/Results/2022/Mar18/syntheticcell/synth.tif"
    #
    # SyntheticCell.standard_synthetic_cell(savepath)
    from src.AnalysisTools import experimentalparams as ep, ShapeMetrics

    printall = False
    savepath = "../../data/temp/"

    cell = SyntheticCell.generatecuboidwithparticle()
    cell = SyntheticCell.generate_synthetic_cell()
    print(cell.shape)
    cuboid = cell[:, 0, :, :, :].squeeze()
    centroid, volume, xspan, yspan, zspan, maxferet, meanferet, minferet, miparea, sphericity = ShapeMetrics.calculate_object_properties(
        cuboid)
    # print("CENTROID: ", centroid/(ep.ZSCALE, ep.XSCALE, ep.YSCALE))
    # print("VOLUME: ",volume/ep.VOLUMESCALE)
    # print("XSPAN: ",xspan/ep.XSCALE)
    # print("YSPAN: ",yspan/ep.YSCALE)
    # print("ZSPAN: ",zspan/ep.ZSCALE)
    # print("MAXFERET: ",maxferet/ep.XSCALE)
    # print("MEANFERET: ",meanferet/ep.XSCALE)
    # print("MINFERET: ", minferet/ep.XSCALE)
    # print("MIPAREA: ", miparea/ep.AREASCALE)
    # print("SPH: ",sphericity)
    particle = cell[:, 1, :, :, :].squeeze()
    organellecounts, centroids, volumes, xspans, yspans, zspans, maxferets, meanferets, minferets, mipareas, orientations3D, z_distributions, radial_distribution2ds, radial_distribution3ds, meanvolume = ShapeMetrics.calculate_multiorganelle_properties(
        particle, centroid)
    print(particle.shape, cuboid.shape)
    d2m, d2s, d2map = ShapeMetrics.distance_from_wall_2d(org_bbox=particle, cell_bbox=cuboid, returnmap=True)
    d3m, d3s, d3map = ShapeMetrics.distance_from_wall_3d(org_bbox=particle, cell_bbox=cuboid, returnmap=True)
    # OmeTiffWriter.save(data=cuboid, uri=savepath + "cuboid.tiff", overwrite_file=True)
    OmeTiffWriter.save(data=cell, uri=savepath + "synthcell.tiff", overwrite_file=True)
    OmeTiffWriter.save(data=d2map, uri=savepath + "d2map.tiff", overwrite_file=True)
    OmeTiffWriter.save(data=d3map, uri=savepath + "d3map.tiff", overwrite_file=True)

    print(d2m, d2s, np.unique(d2map))
    print()
    print()
    print()
    print(d3m, d3s, np.unique(d3map))
    if printall:
        print("\n\nindividual properties")
        print("organellecounts", organellecounts)
        print("centroids", centroids / (ep.ZSCALE, ep.XSCALE, ep.YSCALE))
        print("volumes", volumes / ep.VOLUMESCALE)
        print("xspans", xspans / ep.XSCALE)
        print("yspans", yspans / ep.YSCALE)
        print("zspans", zspans / ep.ZSCALE)
        print("meanferets", meanferets / ep.XSCALE)
        print("minferets", minferets / ep.XSCALE)
        print("maxferets", maxferets / ep.XSCALE)
        print("mipareas", mipareas / ep.AREASCALE)
        print("orientations3D", orientations3D)
        print("z_distributions", z_distributions)
        print("radial_distribution2ds", radial_distribution2ds)
        print("radial_distribution3ds", radial_distribution3ds)
        print("meanvolume", meanvolume / ep.VOLUMESCALE)
        OmeTiffWriter.save(data=cuboid, uri=savepath + "cuboid.tiff", overwrite_file=True)
