

# Megadetector Anleitung 
Die Anleitung des [Megadetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) ist sehr detailier und gut erklärt. Die folgenden Schritte in diesem Dokument dienen als Quick start.

# Quick start
1. Erstelle ein Ordner mit dem Namen "[mein_pfad]\megadetector"
    1. innerhalb dieses Ordner speichere das Megadetecor Modell "md_v5a.0.0". [download](https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt)
    1. erstelle einen Ordner "git"
    1. öffne den conda prompt und navigiere zu [mein_pfad]\git
        1. `cd c:\[mein_pfad]\git`
        1. `git clone https://github.com/ecologize/yolov5/`
        1. `git clone https://github.com/Microsoft/cameratraps`
        1. `git clone https://github.com/Microsoft/ai4eutils`
        1. `cd c:\[mein_pfad]\git\cameratraps`
        1. `conda env create --file environment-detector.yml` (dauer 10min+)
        1. `conda activate cameratraps-detector`
        1. `set PYTHONPATH=%PYTHONPATH%;c:\[mein_pfad]git\cameratraps;c:\[mein_pfad]\git\ai4eutils;c:\[mein_pfad]\git\yolov5`

    Der Pfad muss bei jeder Anwendung erneut gesetzt werden. Der Pythonpfad kann auch [fest](https://www.computerhope.com/issues/ch000549.htm) gesetzt werden. 
1. Im gleichen conda prompt können zwei python scripts ausgeführt werden. "run_detector.py" erhält ein Bild und erstellt ein zweites Bild mit einer Bounding Box. "run_detector_batch.py" wird verwendet um viele Bilder einzulesen und die für ein Postprocessing verwendet werden, die Ausgabedatei ist ein Json file mit der Bounding Box.
    1. Ausführen von "run_detector.py" (cuda:):   
    `python detection\run_detector.py "c:[mein_pfad]\megadetector\md_v5a.0.0.pt" --image_dir "c:[mein_pfad]\megadetector\train_features"  --threshold 0.1`
    1. Ausführen von "run_detector_batch.py" (cuda:36min):  
     `python detection\run_detector_batch.py "c:[mein_pfad]\megadetector\md_v5a.0.0.pt" "c:\Users\manue\megadetector\train_features" c:[mein_pfad]\megadetector\test_output.json" --output_relative_filenames --recursive --checkpoint_frequency 10000 --quiet`

# Labeling und Computing   
To apply this model to larger image sets on a single machine, we recommend a different script, run_detector_batch.py. This outputs data in the same format as our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), so you can leverage all of our post-processing tools. The format that this script produces is also compatible with [Timelapse](https://saul.cpsc.ucalgary.ca/timelapse/).

**Timelapse**      
Wird verwendet um Bilder von Wildtierkameras anzuzeigen und von Hand verschiedene Labels und Metadaten zuzuordnen.   
<img src="Timelapse.jpg" alt="Timelaps, bear-label" style="width: 30%;" />


**batch processing API**   
With the batch processing API, you can process a batch of up to a few million images in one request to the API. If in addition you have some images that are labeled, we can evaluate the performance of the MegaDetector on your labeled images (see [Post-processing tools](https://github.com/microsoft/CameraTraps/tree/main/api/batch_processing#post-processing-tools)).

