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
                new Pair<>(new Point(10.9, -8.20, 5.495), new Quaternion(0f, 0.707f, 0, 0.707f)),
                new Pair<>(new Point(10.925, -10.25, 4.695), new Quaternion(0f, 0f, -0.707f, 0.707f)),
                new Pair<>(new Point(10.925, -10.25, 4.695), new Quaternion(0f, 0f, -0.707f, 0.707f))
        ));

        List<Mat> croppedImages;

        //--------------------------------------------- MISSION START -------------------------------------------------------
        //--------------------------------------------- Area Exploring -------------------------------------------------------

        // The mission starts.
        api.startMission(); // Coordinate -> x:9.815 y:-9.806 z:4.293 | p = 1 0 0 0

        try {
            loadModel();
        } catch (IOException e) {
            e.printStackTrace();
        }

        //-- Move to the first area --

        api.moveTo(areaCenters.get(0).first, areaCenters.get(0).second, false);
        // Take a photo and detect objects
        Mat image = api.getMatNavCam();
        api.saveMatImage(image, "first_area.png");
        api.saveMatImage(undistortedImage(image), "undistorted_first_area.png");
        croppedImages = CroppedContours(image);
        for (Mat region : croppedImages) {
            classifyImage(region);
        }
        //-- Move to the second area --

        api.moveTo(areaCenters.get(1).first, areaCenters.get(1).second, false);
        // Take a photo and detect objects
        image = api.getMatNavCam();
        api.saveMatImage(image, "second_area.png");
        api.saveMatImage(undistortedImage(image), "undistorted_second_area.png");
        croppedImages = CroppedContours(image);


        //-- Move to the third area --
        Point point = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion quaternion = new Quaternion(0f, 0f, 1f, 0f);
        api.moveTo(point, quaternion, false);

        // Take a photo and detect objects
        image = api.getMatNavCam();
        api.saveMatImage(image, "third_area.png");
        api.saveMatImage(undistortedImage(image), "undistorted_third_area.png");
        croppedImages = CroppedContours(image);


        // Move to the second Oasis zone
        // api.moveTo(areaCenters.get(1).first, areaCenters.get(0).second,false);

        // Area scanning.
        // scanArea();


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


    // Model

    private void loadModel() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("cls_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer modelFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        tflite = new Interpreter(modelFile);
    }

    // Predict class for a single image
    private String classifyImage(Mat image) {
        // Step 1: Resize to model input size (e.g., 96x96)
        int inputSize = 96;
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(inputSize, inputSize));

        // Step 2: Convert to RGB
        Mat rgbImage = new Mat();
        Imgproc.cvtColor(resized, rgbImage, Imgproc.COLOR_BGR2RGB);

        // Step 3: Normalize and prepare input [1, 96, 96, 3]
        float[][][][] input = new float[1][inputSize][inputSize][3];
        for (int y = 0; y < inputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                double[] pixel = rgbImage.get(y, x);
                input[0][y][x][0] = (float) (pixel[0] / 255.0);  // R
                input[0][y][x][1] = (float) (pixel[1] / 255.0);  // G
                input[0][y][x][2] = (float) (pixel[2] / 255.0);  // B
            }
        }

        // Step 4: Prepare output buffer for [1, num_classes]
        float[][] output = new float[1][12]; // You have 12 classes (including "unwanted")

        // Step 5: Run inference
        tflite.run(input, output);

        // Step 6: Get top-1 prediction
        float maxProb = -1f;
        int maxIndex = -1;
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }

        String predictedClass = getClassName(maxIndex);
        System.out.println("Predicted: " + predictedClass + " (" + maxProb + ")");
        return predictedClass;
    }

    // Map class ID to label
    private String getClassName(int id) {
        String[] classes = {
                "coin", "compass", "coral", "crystal", "diamond",
                "emerald", "fossil", "key", "letter", "shell", "treasure_box", "unwanted"
        };
        return (id >= 0 && id < classes.length) ? classes[id] : "Unknown";
    }

    // Image Processing

    private Mat undistortedImage(Mat inputImage) {
        // Resize
        Mat resized = new Mat();
        Imgproc.resize(inputImage, resized, new Size(1280, 960));

        // Undistort
        Mat undistorted = new Mat();

        Mat K = new Mat(3, 3, CvType.CV_64F);
        K.put(0, 0,
                523.105750, 0.000000, 635.434258,
                0.000000, 534.765913, 500.335102,
                0.000000, 0.000000, 1.000000
        );

        MatOfDouble distCoeffs = new MatOfDouble(
                -0.164787, 0.020375, -0.001572, -0.000369, 0.0
        );

        // Prepare ROI container
        Rect roi = new Rect();
        Mat newK = Calib3d.getOptimalNewCameraMatrix(K, distCoeffs, new Size(1280, 960), 1, new Size(1280, 960), roi);

        // Undistort the image
        Calib3d.undistort(resized, undistorted, K, distCoeffs, newK);

        // Crop to valid region
        Mat validArea = new Mat(undistorted, roi);

        return validArea;
    }


    private List<Mat> CroppedContours(Mat inputImage) {

        List<Mat> significantRegions = new ArrayList<>();

        Mat undistorted = undistortedImage(inputImage);
        Mat gray = undistorted.clone();

        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        DetectorParameters parameters = DetectorParameters.create();
        Aruco.detectMarkers(gray, dictionary, corners, ids, parameters);

        if (ids.empty()) {
            System.out.println("No AR tag detected.");
            return significantRegions; // return empty
        }

        for (int i = 0; i < ids.rows(); i++) {
            Mat points = corners.get(i);
            org.opencv.core.Point[] contourPoints = new org.opencv.core.Point[(int) points.total()];

            for (int j = 0; j < points.total(); j++) {
                double[] coords = points.get(0, j);
                contourPoints[j] = new org.opencv.core.Point(coords[0], coords[1]);
            }

            MatOfPoint matOfPoint = new MatOfPoint(contourPoints);
            Rect rect = Imgproc.boundingRect(matOfPoint);

            int pad = 250;
            int x1 = Math.max(rect.x - pad, 0);
            int y1 = Math.max(rect.y - pad, 0);
            int x2 = Math.min(rect.x + rect.width + pad, gray.width());
            int y2 = Math.min(rect.y + rect.height + pad, gray.height());

            Rect paddedRect = new Rect(x1, y1, x2 - x1, y2 - y1);
            Mat tagCrop = new Mat(gray, paddedRect);

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

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(closed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            for (int j = 0; j < contours.size(); j++) {
                MatOfPoint cnt = contours.get(j);
                double area = Imgproc.contourArea(cnt);

                if (area < 100 || !isClosedContour(cnt)) continue;

                Rect contourRect = Imgproc.boundingRect(cnt);
                if (contourRect.width < 10 || contourRect.height < 10) continue;

                int cx1 = Math.max(contourRect.x - 5, 0);
                int cy1 = Math.max(contourRect.y - 5, 0);
                int cx2 = Math.min(contourRect.x + contourRect.width + 5, tagCrop.width());
                int cy2 = Math.min(contourRect.y + contourRect.height + 5, tagCrop.height());

                if (cx2 > cx1 && cy2 > cy1) {
                    Mat croppedRegion = new Mat(tagCrop, new Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
                    significantRegions.add(croppedRegion);
                }
            }
        }

        return significantRegions;
    }

    // Utility method
    private boolean isClosedContour(MatOfPoint cnt) {
        org.opencv.core.Point[] pts = cnt.toArray();
        if (pts.length < 2) return false;
        double distance = Math.sqrt(Math.pow(pts[0].x - pts[pts.length - 1].x, 2) + Math.pow(pts[0].y - pts[pts.length - 1].y, 2));
        return distance < 10;
    }


}