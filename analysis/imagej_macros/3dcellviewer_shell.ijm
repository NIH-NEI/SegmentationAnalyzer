individualcellstackpath = "C:/Users/satheps/PycharmProjects/Results/2022/Mar4/channels/lamp1/results/cellstacks";
savepath = "C:/Users/satheps/PycharmProjects/Results/2022/Mar4/channels/lamp1/results/imagejcellstacks";
cellstacklist  = getFileList(individualcellstackpath);

print(cellstacklist.length)
for (i=0; i<cellstacklist.length; i++) {
	if (cellstacklist[i].contains(".tiff")){
		cellstackpath = individualcellstackpath+ File.separator +cellstacklist[i];
		basename = split(cellstacklist[i],".");//[0];
		basename = basename[0];
//		print(cellstacklist[i]);
		run("Bio-Formats Importer", "open=" + cellstackpath + " autoscale color_mode=Composite rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
		print(getTitle());
		selectWindow(""+cellstacklist[i]+" - C=0");
		/*Use outline if the original cell borders are solid instead of shell-like*/
//		run("Outline", "stack");
		
		run("Merge Channels...", "c1=["+cellstacklist[i]+" - C=0] c2=["+cellstacklist[i]+" - C=1] c5=["+cellstacklist[i]+" - C=2] create");
		selectWindow("Composite");
		getDimensions(width, height, channels, slices, frames);
		run("Properties...", "channels="+channels+" slices="+slices+" frames="+frames+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=2.3077");
		saveAs("Tiff", savepath+File.separator+ "Composite_"+cellstacklist[i]);
//		saveAs("gif", savepath+File.separator+ "Composite_"+cellstacklist[i]);
		close("*");
	}
}
