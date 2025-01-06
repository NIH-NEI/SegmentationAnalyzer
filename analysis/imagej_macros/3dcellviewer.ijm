individualcellstackpath = "C:/Users/satheps/PycharmProjects/Results/2022/Feb11/TOM/results_test2";
savepath = "C:/Users/satheps/PycharmProjects/Results/2022/Feb11/TOM/results_test2/overlays";
cellstacklist  = getFileList(individualcellstackpath);


for (i=0; i<cellstacklist.length; i++) {
	if (cellstacklist[i].contains(".tiff")){
		cellstackpath = individualcellstackpath+ File.separator +cellstacklist[i];
		basename = split(cellstacklist[i],".");//[0];
		basename = basename[0];
		run("Bio-Formats Importer", "open=" + cellstackpath + " autoscale color_mode=Composite rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
		run("Merge Channels...", "c1=["+cellstacklist[i]+" - C=0] c2=["+cellstacklist[i]+" - C=1] c5=["+cellstacklist[i]+" - C=2] create");
		selectWindow("Composite");
		getDimensions(width, height, channels, slices, frames);
		run("Properties...", "channels="+channels+" slices="+slices+" frames="+frames+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=2.3077");
		saveAs("Tiff", savepath+File.separator+ "Composite_"+cellstacklist[i]);
		saveAs("gif", savepath+File.separator+ "Composite_"+cellstacklist[i]);
//		waitForUser;

//		run("3D Viewer");
//		call("ij3d.ImageJ3DViewer.setCoordinateSystem", "false");
//		call("ij3d.ImageJ3DViewer.setTransform", "-0.5782809 0.8155682 0.020969868 39.77776 -0.4780244 -0.35954973 0.8013842 94.95529 0.6611232 0.4534011 0.5977822 -57.690453 0.0 0.0 0.0 1.0 ");
//		saveAs("Jpeg", savepath+File.separator +"Composite3d_"+cellstacklist[i]);
//		call("ij3d.ImageJ3DViewer.add", "composite_"+cellstacklist[i], "None", "compositeTOM20_P1-W3-TOM_E02_F002_132.npz.tif", "0", "true", "true", "true", "1", "0");
//		call("ij3d.ImageJ3DViewer.select", "composite_"+cellstacklist[i]);
	//	saveAs("Jpeg", individualcellstackpath+File.separator+ "view1_composite_"+cellstacklist[i]);
//		selectWindow("Snapshot");
//		call("ij3d.ImageJ3DViewer.snapshot", "512", "512");
//		saveAs("Jpeg", individualcellstackpath+File.separator+ "view1_composite_"+cellstacklist[i]+".jpg");
		close("*");
	}
}
