package jp.jaxa.iss.kibo.rpc.sampleapk;

        import android.util.Pair;

        import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

        import gov.nasa.arc.astrobee.types.Point;
        import gov.nasa.arc.astrobee.types.Quaternion;

        import org.opencv.core.Mat;

        import java.io.IOException;
        import java.util.ArrayList;
        import java.util.Arrays;
        import java.util.HashMap;

        import org.tensorflow.lite.Interpreter;

        import java.io.FileInputStream;

        import java.nio.MappedByteBuffer;
        import java.nio.channels.FileChannel;

        import org.opencv.core.Size;

        import org.opencv.imgproc.Imgproc;
        import android.content.res.AssetFileDescriptor;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {

    private Interpreter tflite;

    @Override
    protected void runPlan1(){
        //--------------------------------------------- Variables Declaration -------------------------------------------------------

        HashMap<String, ArrayList<String>> foundItemsMap = new HashMap<>();
        String currentArea;
        String foundItem;
        int id;
        ArrayList<String> itemsArr = new ArrayList<>();
        ArrayList<Pair<Point, Quaternion>> areaCenters = new ArrayList<>(Arrays.asList(
                new Pair<>(new Point(10.925, -10.00, 4.695), new Quaternion(0f, 0f, -0.707f, 0.707f)),
                new Pair<>(new Point(11.175, -8.875, 5.195), new Quaternion(0f, 0f, 0, 0)),
                new Pair<>(new Point(10.925, -10.25, 4.695), new Quaternion(0f, 0f, -0.707f, 0.707f)),
                new Pair<>(new Point(10.925, -10.25, 4.695), new Quaternion(0f, 0f, -0.707f, 0.707f))
        ));

        //--------------------------------------------- MISSION START -------------------------------------------------------
        //--------------------------------------------- Area Exploring -------------------------------------------------------

        // The mission starts.
        api.startMission(); // Coordinate -> x:9.815 y:-9.806 z:4.293 | p = 1 0 0 0

        try {
            loadModel();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Move to the first area
        api.moveTo(areaCenters.get(0).first, areaCenters.get(0).second, false);

        // Take a photo and detect objects
        Mat image = api.getMatNavCam();
        ArrayList<String> detectedItems = detectObjects(image);

        // Log detected items
        for (String item : detectedItems) {

            System.out.println("Detected: " + item);
        }



        // Move to the second Oasis zone
//        api.moveTo(areaCenters.get(1).first, areaCenters.get(0).second,false);

        // Area scanning.
//        scanArea();


        /* ******************************************************************************** */

        api.reportRoundingCompletion();

        //------------------------------------------------- Treasure Finding ---------------------------------------------------

        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();
    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }


    private int readAR(Mat image){

        // Use ArUco detector (part of OpenCV contrib modules)
        org.opencv.aruco.Dictionary dictionary = org.opencv.aruco.Aruco.getPredefinedDictionary(org.opencv.aruco.Aruco.DICT_5X5_250);
        java.util.List<Mat> corners = new java.util.ArrayList<>();
        Mat ids = new Mat();
        org.opencv.aruco.DetectorParameters parameters = org.opencv.aruco.DetectorParameters.create();
        org.opencv.aruco.Aruco.detectMarkers(image, dictionary, corners, ids, parameters);

        System.out.println("Raw data from AR tag:" + ids);

        // Log AR tag IDs
        if (!ids.empty()) {
            for (int i = 0; i < ids.rows(); i++) {
                int id = (int) ids.get(i, 0)[0];
                System.out.println("Detected AR tag ID: " + id);
                return id;
            }
        }

        System.out.println("AR tag wasn't detected!!!");
        return -1;
    }


    private void scanArea(Mat image){

        //pre-processing

        int id = readAR(image);
    }

    private void readImage(){

    }

    // Model


    // Load model from assets
    private void loadModel() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("best_float32.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer modelFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        tflite = new Interpreter(modelFile);
    }

    private ArrayList<String> detectObjects(Mat image) {
        // Preprocess image
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(640, 640));

        // Convert to RGB if needed (YOLOv8 expects RGB)
        Mat rgbImage = new Mat();
        Imgproc.cvtColor(resized, rgbImage, Imgproc.COLOR_BGR2RGB);

        // Convert to float array (input tensor)
        float[][][][] input = new float[1][640][640][3];
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                double[] pixel = rgbImage.get(y, x);
                input[0][y][x][0] = (float) (pixel[0] / 255.0);
                input[0][y][x][1] = (float) (pixel[1] / 255.0);
                input[0][y][x][2] = (float) (pixel[2] / 255.0);
            }
        }

        // YOLOv8 output format: [batch, num_outputs, num_classes+4]
        // For 11 classes: [1, 8400, 15] (11 classes + 4 bbox coords)
        float[][][] output = new float[1][15][8400];

        // Run inference
        tflite.run(input, output);

        // Process results
        ArrayList<String> detections = new ArrayList<>();
        float confidenceThreshold = 0.5f;

        if (detections.isEmpty()) {
            System.out.println("No objects detected in the image.");
        }

        for (int i = 0; i < output[0].length; i++) {
            // Find highest class confidence
            int bestClassId = -1;
            float bestConf = 0;

            for (int c = 0; c < 11; c++) {  // Iterate through 11 classes
                if (output[0][i][c+4] > bestConf) {
                    bestConf = output[0][i][c+4];
                    bestClassId = c;
                }
            }

            // If confidence exceeds threshold, add to detections
            if (bestConf > confidenceThreshold && bestClassId >= 0) {
                // Get bounding box coordinates (x,y,w,h)
                float x = output[0][i][0];
                float y = output[0][i][1];
                float w = output[0][i][2];
                float h = output[0][i][3];

                detections.add(getClassName(bestClassId) +
                        " (" + bestConf + ") at [" + x + "," + y + "]");
            }
        }



        return detections;
    }

    private String getClassName(int id) {
        // Replace with your 11 actual class names
        String[] classes = {
                "Class1", "Class2", "Class3", "Class4", "Class5",
                "Class6", "Class7", "Class8", "Class9", "Class10", "Class11"
        };
        return (id >= 0 && id < classes.length) ? classes[id] : "Unknown";
    }

}