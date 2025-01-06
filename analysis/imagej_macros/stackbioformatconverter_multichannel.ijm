

//channelnames = ['ACTB', 'CETN2', 'CTNNB1', 'DSP', 'FBL', 'GJA1', 'LAMP1', 'LC3B', 'MYH', 'RAB5', 'Sec61', 'SLC', 'ST6GAL1', 'TOM20', 'ZO1'];
mainDir = "C:/Users/satheps/PycharmProjects/Results/2022/Imaris visualizations/";
channelnames = getFileList(mainDir); 

for(c=0; c<channelnames.length; c++){
	channelname = channelnames[c];
	print(channelname);
	path_segmentedstacks = "C:/Users/satheps/PycharmProjects/Results/2022/Imaris visualizations/"+channelname+"imgs/";
	path_bioformatted = "C:/Users/satheps/PycharmProjects/Results/2022/Imaris visualizations/"+channelname+"bioformat_stacks/";
	//
	//path_segmentedstacks = "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/";
	//path_bioformatted = "C:/Users/satheps/PycharmProjects/Results/2022/Mar25/illustrations_TOM/OVERLAY/";
	
	subdirfilelist_segmentedstacks = getFileList(path_segmentedstacks);
	Array.print(subdirfilelist_segmentedstacks);
	
	for (i=0; i<subdirfilelist_segmentedstacks.length; i++) {
		print("Starting " + subdirfilelist_segmentedstacks[i]);
		currentfilepath = path_segmentedstacks +subdirfilelist_segmentedstacks[i];
		if(subdirfilelist_segmentedstacks[i].contains(".tiff")){
			print(currentfilepath);
			run("Bio-Formats Importer", "open="+currentfilepath+" autoscale color_mode=Composite rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
			run("Merge Channels...", "c1=["+subdirfilelist_segmentedstacks[i]+" - C=0] c2=["+subdirfilelist_segmentedstacks[i]+" - C=1] c3=["+subdirfilelist_segmentedstacks[i]+" - C=2] create");
			selectWindow("Composite");
			File.makeDirectory(path_bioformatted); 
			combinedfilepath = path_bioformatted +subdirfilelist_segmentedstacks[i];
			print("Saving: ", combinedfilepath);
			saveAs("Tiff", combinedfilepath);
			
			close();
		}
	}
}