package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.util.Pair;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.support.image.TensorImage;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {

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


        // Move to the first area
        api.moveTo(areaCenters.get(0).first, areaCenters.get(0).second, false);

        loadModel();

        Bitmap bitmap = api.getBitmapNavCam();

        TensorImage tensorImage = preprocess(bitmap);

        List<Detection> results = objectDetector.detect(tensorImage);

        for (Detection detection : results) {
            System.out.println("Detected class: " + detection.getCategories().get(0).getLabel());
            System.out.println("Confidence: " + detection.getCategories().get(0).getScore());
            System.out.println("BoundingBox: " + detection.getBoundingBox().toString());
        }

        // Move to the second Oasis zone
//        api.moveTo(areaCenters.get(1).first, areaCenters.get(0).second,false);

        // Area scanning.
//        scanArea();

//        // Move to area #2
//        Point area2Center = new Point(10.6, -8.2, 5.2); // Center of Area 1
//        Quaternion lookingAtArea2 = new Quaternion(0, 0, -0.707f, 0.707f); // Facing Area 2
//        api.moveTo(area2Center, lookingAtArea2, false);

        // Get a camera image and read the AR tag.
//        ReadAR();
//
//        // Move to area #3
//        Point area3Center = new Point(10.6, -7.0, 5.1); // Center of Area 1
//        Quaternion lookingAtArea3 = new Quaternion(0, 0, -0.707f, 0.707f); // Facing Area 3
//        api.moveTo(area3Center, lookingAtArea3, false);
//
//        // Get a camera image and read the AR tag.
//        ReadAR();
//
//        // Move to area #4
//        Point area4Center = new Point(10.7, -6.0, 5.2); // Center of Area 1
//        Quaternion lookingAtArea4 = new Quaternion(0, 0, -0.707f, 0.707f); // Facing Area 4
//        api.moveTo(area4Center, lookingAtArea4, false);
//
//        // Get a camera image and read the AR tag.
//        ReadAR();

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // When you recognize landmark items, let’s set the type and number.
//        api.setAreaInfo(1, "item_name", 1);

        /* **************************************************** */
        /* Let's move to each area and recognize the items. */
        /* **************************************************** */

        // When you move to the front of the astronaut, report the rounding completion. ( pos is good )
//        Point point = new Point(11.143d, -6.7607d, 4.9654d);
//        Quaternion quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
//        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();

        //------------------------------------------------- Treasure Finding ---------------------------------------------------
        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */
        /* ********************************************************** */

        //Santi.AI

        // Let's notify the astronaut when you recognize it.
        api.notifyRecognitionItem();

        /* ******************************************************************************************************* */
        /* Write your code to move Astrobee to the location of the target item (what the astronaut is looking for) */
        /* ******************************************************************************************************* */
        // Take a snapshot of the target item.
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

    private ObjectDetector objectDetector;

    // Load model
    private void loadModel() {
        try {
            ObjectDetectorOptions options = ObjectDetectorOptions.builder()
                    .setMaxResults(5)   // Max 5 detections
                    .setScoreThreshold(0.5f) // Confidence threshold
                    .build();

            objectDetector = ObjectDetector.createFromFileAndOptions(this, "best_float32.tflite", options);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private TensorImage preprocess(Bitmap bitmap) {
        // Create a TensorImage object
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        // Load the bitmap into tensorImage
        tensorImage.load(bitmap);

        // Manually normalize the image (divide pixel values by 255)
        TensorImage normalizedImage = TensorImage.fromBitmap(bitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new NormalizeOp(0, 255))  // 0 mean, divide by 255 std
                .build();

        normalizedImage = imageProcessor.process(normalizedImage);

        return normalizedImage;
    }
}