path_segmentedstacks = "C:/Users/satheps/PycharmProjects/Results/2022/Apr29/lc3b/illustrations_LC3B/imgs/";
path_bioformatted = "C:/Users/satheps/PycharmProjects/Results/2022/Apr29/lc3b/illustrations_LC3B/combinedstacks/";
//
//path_segmentedstacks = "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/";
//path_bioformatted = "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/OVERLAY/";

subdirfilelist_segmentedstacks = getFileList(path_segmentedstacks);
Array.print(subdirfilelist_segmentedstacks);

for (i=0; i<subdirfilelist_segmentedstacks.length; i++) {
	print("Starting" + subdirfilelist_segmentedstacks[i]);
	currentfilepath = path_segmentedstacks+ File.separator +subdirfilelist_segmentedstacks[i];
	if(subdirfilelist_segmentedstacks[i].contains(".tiff")){
		run("Bio-Formats Importer", "open="+currentfilepath+" autoscale color_mode=Composite rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
		run("Merge Channels...", "c1=["+subdirfilelist_segmentedstacks[i]+" - C=0] c2=["+subdirfilelist_segmentedstacks[i]+" - C=1] c3=["+subdirfilelist_segmentedstacks[i]+" - C=2] create");
		selectWindow("Composite");
		combinedfilepath = path_bioformatted +subdirfilelist_segmentedstacks[i];
		print("Saving: ", combinedfilepath);
		saveAs("Tiff", combinedfilepath);
		
		close();
	}
}
