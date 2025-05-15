// === Step 1: Select main folder ===
mainFolder = getDirectory("Select the main folder");
subdirs = getFileList(mainFolder);

// === Step 2: Process .nd2 files in main folder ===
for (i = 0; i < subdirs.length; i++) {
    if (endsWith(subdirs[i], ".nd2")) {
        fullPath = mainFolder + subdirs[i];
        run("Bio-Formats Importer", "open=[" + fullPath + "] use=Bio-Formats");

        if (isOpen(subdirs[i])) {
            run("Stack to Images");

            baseName = substring(subdirs[i], 0, lastIndexOf(subdirs[i], "."));
            outputFolder = mainFolder + baseName + "\\";
            File.makeDirectory(outputFolder);

            imageTitles = getList("image.titles");
            for (j = 0; j < imageTitles.length; j++) {
                selectWindow(imageTitles[j]);
                saveAs("Tiff", outputFolder + imageTitles[j] + ".tif");
                close();
            }
        }
    }
}

// === Step 3: Process 'ACC-0001.tif' in all subdirectories ===
subdirs = getFileList(mainFolder);
for (i = 0; i < subdirs.length; i++) {
    subPath = mainFolder + subdirs[i];
    if (File.isDirectory(subPath)) {
        accPath = subPath + "\\ACC-0001.tif";
        if (File.exists(accPath)) {
            open(accPath);

            if (nImages() > 0) {
                setAutoThreshold("Default no-reset");
                setThreshold(3000, 65535, "raw");
                setOption("BlackBackground", true);
                run("Convert to Mask");
                
                run("Analyze Particles...", "size=100-Infinity pixel include add");

                if (roiManager("count") > 0) {
                    open(accPath);
                    setForegroundColor(0, 0, 0);
                    setBackgroundColor(255, 255, 255);
                    roiManager("Select All");
                    roiManager("Show All");
                    roiManager("Fill");
                    saveAs("Tiff", subPath + "\\ACC-0001-fill.tif");
                } else {
                    open(accPath);
                    saveAs("Tiff", subPath + "\\ACC-0001-fill.tif");
                    print("No particles detected in " + subdirs[i] + ", original image saved.");
                }
            } else {
                print("Image failed to open in " + subdirs[i]);
            }

            roiManager("Deselect");
            roiManager("Reset");
            run("Close All");
        }
    }
}

print("All processing completed.");
