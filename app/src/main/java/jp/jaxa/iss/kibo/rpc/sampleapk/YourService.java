package jp.jaxa.iss.kibo.rpc.sampleapk;

        import android.util.Pair;

        import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

        import gov.nasa.arc.astrobee.types.Point;
        import gov.nasa.arc.astrobee.types.Quaternion;

        import org.opencv.aruco.Aruco;
        import org.opencv.aruco.DetectorParameters;
        import org.opencv.aruco.Dictionary;
        import org.opencv.calib3d.Calib3d;
        import org.opencv.core.CvType;
        import org.opencv.core.Mat;

        import java.io.IOException;
        import java.util.ArrayList;
        import java.util.Arrays;
        import java.util.HashMap;

        import org.opencv.core.MatOfDouble;
        import org.opencv.core.MatOfPoint;
        import org.opencv.core.Rect;
        import org.opencv.imgproc.CLAHE;
        import org.tensorflow.lite.Interpreter;

        import java.io.FileInputStream;

        import java.nio.MappedByteBuffer;
        import java.nio.channels.FileChannel;
        import java.util.List;

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
                new Pair<>(new Point(11.1, -10.00, 5.25), new Quaternion(0f, 0f, -0.707f, 0.707f)),
                new Pair<>(new Point(11.175, -8.875, 5.195), new Quaternion(0f, -0.707f, 0, 0.707f)),
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
        api.saveMatImage(image, "first_area_4");
        processImageOnAstrobee(image);

        ArrayList<String> detectedItems = detectObjects(image);

        // Log detected items
        for (String item : detectedItems) {

            System.out.println("Detected: " + item);
        }

        // Move to the first area
        api.moveTo(areaCenters.get(1).first, areaCenters.get(1).second, false);

        // Take a photo and detect objects
        image = api.getMatNavCam();
        api.saveMatImage(image, "second_area");
        detectedItems = detectObjects(image);

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

            System.out.println(getClassName(bestClassId));


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


    private void processImageOnAstrobee(Mat inputImage) {
        // Resize
        Mat resized = new Mat();
        Imgproc.resize(inputImage, resized, new Size(1280, 960));

        // Undistort
        Mat undistorted = new Mat();
        Mat K = new Mat(3, 3, CvType.CV_64F);
        K.put(0, 0,
                523.105750, 0.000000, 635.434258,
                0.000000, 534.765913, 500.335102,
                0.000000, 0.000000, 1.000000);

        Mat distCoeffs = new MatOfDouble(
                -0.164787, 0.020375, -0.001572, -0.000369, 0.0
        );

        Mat newK = Calib3d.getOptimalNewCameraMatrix(K, distCoeffs, new Size(1280, 960), 1, new Size(1280, 960), null);
        Calib3d.undistort(resized, undistorted, K, distCoeffs, newK);

        // Detect ARUCO
        Mat gray = undistorted.clone(); // Already grayscale if from NavCam
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        DetectorParameters parameters = DetectorParameters.create();
        Aruco.detectMarkers(gray, dictionary, corners, ids, parameters);

        if (ids.empty()) {
            System.out.println("No AR tag detected.");
            return;
        }

        for (int i = 0; i < ids.rows(); i++) {
            Mat markerCorners = corners.get(i);

            // Find bounding box
            MatOfPoint matOfPoint = new MatOfPoint(markerCorners);
            Rect boundingRect = Imgproc.boundingRect(matOfPoint);

            int pad = 250;
            int x1 = Math.max(boundingRect.x - pad, 0);
            int y1 = Math.max(boundingRect.y - pad, 0);
            int x2 = Math.min(boundingRect.x + boundingRect.width + pad, gray.width());
            int y2 = Math.min(boundingRect.y + boundingRect.height + pad, gray.height());

            Mat tagCrop = new Mat(gray, new Rect(x1, y1, x2 - x1, y2 - y1));

            // CLAHE Enhancement
            CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
            Mat enhanced = new Mat();
            clahe.apply(tagCrop, enhanced);

            // Gaussian Blur
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(enhanced, blurred, new Size(5, 5), 0);

            // Canny Edge Detection
            Mat edges = new Mat();
            Imgproc.Canny(blurred, edges, 200, 400);

            // Morphological Closing
            Mat closed = new Mat();
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel);

            // Find Contours
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(closed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            for (int j = 0; j < contours.size(); j++) {
                MatOfPoint cnt = contours.get(j);
                double area = Imgproc.contourArea(cnt);

                if (area < 100) continue;
                if (!isClosedContour(cnt)) continue;

                Rect rect = Imgproc.boundingRect(cnt);
                if (rect.width < 10 || rect.height < 10) continue;

                int cx1 = Math.max(rect.x - 5, 0);
                int cy1 = Math.max(rect.y - 5, 0);
                int cx2 = Math.min(rect.x + rect.width + 5, tagCrop.width());
                int cy2 = Math.min(rect.y + rect.height + 5, tagCrop.height());

                Mat croppedRegion = new Mat(tagCrop, new Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
                // You can now save or analyze 'croppedRegion'
                String filename = "contour_" + j + ".png";
                api.saveMatImage(croppedRegion, filename);
                detectObjects(croppedRegion);
                System.out.println("Saved: " + filename);
            }
        }
    }

    // Utility method
    private boolean isClosedContour(MatOfPoint cnt) {
        org.opencv.core.Point[] pts = cnt.toArray();
        if (pts.length < 2) return false;
        double distance = Math.sqrt(Math.pow(pts[0].x - pts[pts.length - 1].x, 2) + Math.pow(pts[0].y - pts[pts.length - 1].y, 2));
        return distance < 10;
    }


}