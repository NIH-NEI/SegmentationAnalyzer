import os
from analysis.stackio import stackio
import numpy as np

pathd = "..data/../LC3B_Comparison/calcs/"
pathb = "..data/../LC3B_Comparison/calcs_biowulf/"
pathb_nomp = "..data/../LC3B_Comparison/calcs_nomp/"
pathbe = "..data/../LC3B_Comparison/calcs_hram/"

fbiowulf = [f for f in os.listdir(pathb) if f.__contains__("npz")]
fdesktop = [f for f in os.listdir(pathd) if f.__contains__("npz")]
fbionomp = [f for f in os.listdir(pathd) if f.__contains__("npz")]
fbioeram = [f for f in os.listdir(pathd) if f.__contains__("npz")]

print(fbiowulf)
print(fdesktop)
for fb in fbiowulf:
    for fd in fdesktop:
        for fbn in fbionomp:
            for fbe in fbioeram:
                if fb == fd == fbn == fbe:
                    fbp = os.path.join(pathb, fb)
                    fdp = os.path.join(pathd, fd)
                    fbnp = os.path.join(pathd, fbn)
                    fbep = os.path.join(pathd, fbe)
                    loadedb = stackio.loadproperty(fbp)
                    loadedd = stackio.loadproperty(fdp)
                    loadedbnp = stackio.loadproperty(fbnp)
                    loadedbep = stackio.loadproperty(fbep)
                    success = stackio.checksavedfileintegrity(loadedb, loadedd)
                    success2 = stackio.checksavedfileintegrity(loadedbnp, loadedd)
                    success3 = stackio.checksavedfileintegrity(loadedbep, loadedd)
                    print(f"\n{fb} == {fd}: {success}\t{fbn} == {fd}: {success}\t{fbe} == {fd}: {success}\t")
                    print(f"nan_count: {np.count_nonzero(np.isnan(loadedb))} : {np.count_nonzero(np.isnan(loadedd))}")
                    print(f"nan_count: {np.count_nonzero(np.isnan(loadedbnp))} : {np.count_nonzero(np.isnan(loadedd))}")
                    print(f"nan_count: {np.count_nonzero(np.isnan(loadedbep))} : {np.count_nonzero(np.isnan(loadedd))}")
                    print(
                        f"nan_mean: {np.nanmean(loadedb)} : {np.nanmean(loadedd)}, diff = {np.nanmean(loadedd) - np.nanmean(loadedb)}")
                    print(
                        f"nan_mean: {np.nanmean(loadedbnp)} : {np.nanmean(loadedd)}, diff = {np.nanmean(loadedd) - np.nanmean(loadedbnp)}")
                    print(
                        f"nan_mean: {np.nanmean(loadedbep)} : {np.nanmean(loadedd)}, diff = {np.nanmean(loadedd) - np.nanmean(loadedbep)}")
                    print(
                        f"nan_std: {np.nanstd(loadedb)} : {np.nanstd(loadedd)}, diff = {np.nanstd(loadedd) - np.nanstd(loadedb)}")
                    print(
                        f"nan_std: {np.nanstd(loadedbnp)} : {np.nanstd(loadedd)}, diff = {np.nanstd(loadedd) - np.nanstd(loadedbnp)}")
                    print(
                        f"nan_std: {np.nanstd(loadedbep)} : {np.nanstd(loadedd)}, diff = {np.nanstd(loadedd) - np.nanstd(loadedbep)}")
# os.path.join(path2, f)
# loaded = stackio.loadproperty(fbiowulf)
# success = stackio.checksavedfileintegrity(loaded, prop)
