// Let the user select a folder
dir = getDirectory("Select a folder");

// Get all TIFF files in the folder
list = getFileList(dir);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {
        // Open the file
        open(dir + list[i]);

        run("8-bit");

        // Set foreground and background colors
        setForegroundColor(255, 255, 255);
        setBackgroundColor(0, 0, 0);

        // Fill four corner rectangles (140x140) across all slices
        makeRectangle(0, 0, 140, 140);
        run("Fill", "stack");
        makeRectangle(160, 0, 140, 140);
        run("Fill", "stack");
        makeRectangle(160, 160, 140, 140);
        run("Fill", "stack");
        makeRectangle(0, 160, 140, 140);
        run("Fill", "stack");

        // Reset selection to entire image
        makeRectangle(0, 0, 300, 300);

        // Apply threshold
        setThreshold(205, 255, "raw");
        setOption("BlackBackground", true);
        run("Convert to Mask");

        // Invert the image stack
        run("Invert", "stack");

        // Analyze particles (size: 200-Infinity) and add to ROI manager
        run("Analyze Particles...", "size=200-Infinity add stack");

        // Measure all ROIs
        roiManager("Measure");

        // Save the results to CSV (same name as the image)
        saveAs("Results", dir + replace(list[i], ".tif", ".csv"));

        // Clean up
        run("Clear Results");
        roiManager("Reset");
        close();
    }
}

print("✅ Processing complete!");

