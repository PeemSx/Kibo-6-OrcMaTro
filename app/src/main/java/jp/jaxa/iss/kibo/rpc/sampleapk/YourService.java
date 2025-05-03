package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Pair;

import gov.nasa.arc.astrobee.Kinematics;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.json.JSONObject;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

        List<Map<String, Integer>> foundItemsPerArea = new ArrayList<>();
        List<Mat> areaImages = new ArrayList<>();
        int id;
        Point dst;
        ArrayList<Pair<Point, Quaternion>> areaCenters = new ArrayList<>(Arrays.asList(
                new Pair<>(new Point(11.0, -10.00, 5.25), new Quaternion(0f, 0f, -0.707f, 0.707f)),
                new Pair<>(new Point(10.9, -8.75, 4.4), new Quaternion(0f, 0.707f, 0, 0.707f)),
                new Pair<>(new Point(10.9, -7.25, 4.4), new Quaternion(0f, 0.707f, 0, 0.707f)),
                new Pair<>(new Point(10.6, -6.76, 4.96), new Quaternion(0f, 0f, 1, 0))
        ));
//        ArrayList<Pair<Point, Quaternion>> oasisCenters = new ArrayList<>(Arrays.asList(
//                new Pair<>(
//                        new Point(10.925, -9.85, 4.695),      //  (10.425+11.425)/2 , (-10.2-9.5)/2 , (4.445+4.945)/2
//                        new Quaternion(0f, 0f, -0.707f, 0.707f)
//                ),
//                new Pair<>(
//                        new Point(11.175, -8.975, 5.195),      //  (10.925+11.425)/2 , (-9.5-8.45)/2 , (4.945+5.445)/2
//                        new Quaternion(0f, 0.707f, 0f, 0.707f)
//                ),
//
//                new Pair<>(
//                        new Point(10.700, -7.925, 5.195),      //  (10.425+10.975)/2 , (-8.45-7.4)/2 , (4.945+5.445)/2
//                        new Quaternion(0f, 0f, -0.707f, 0.707f)
//                ),
//
//                new Pair<>(
//                        new Point(11.175, -6.875, 4.685),      //  (10.925+11.425)/2 , (-7.4-6.35)/2 , (4.425+4.945)/2
//                        new Quaternion(0f, 0.707f, 0f, 0.707f)
//                )
//        ));
        Mat image;
        double[][] cropParams = {
                {0.0, 0.0, 0.0, 0.1},  // Area 1
                {0.0, 0.0, 0.20, 0.20}, // Area 2
                {0.0, 0.0, 0.0, 0.50},  // Area 3
                {0.2, 0.0, 0.2, 0.4}   // Area 4
        };
        //--------------------------------------------- MISSION START -------------------------------------------------------
        //--------------------------------------------- Area Exploring -------------------------------------------------------

        // The mission starts.
        api.startMission(); // Coordinate -> x:9.815 y:-9.806 z:4.293 | p = 1 0 0 0

        try {
            loadModel();
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < 4; i++) {
            moveToWithCheck(areaCenters.get(i).first, areaCenters.get(i).second, false);
            // Take a picture of the area
            image = api.getMatNavCam();
            id = readAR(image);
            api.saveMatImage(image, "area_" + id + ".png");
            // re-take the picture if the id is incorrect
            if (id < 0 || id > 4) {
                image = api.getMatNavCam();
                id = readAR(image);
            }

            // move and rotate using AR tag's info
            Pair<Point, Quaternion> goal = computeTagApproachPose(image);
            if (goal != null) {
                if(i == 4){
                    dst = new Point(areaCenters.get(i).first.getX(), goal.first.getY(), goal.first.getZ());
                }else if (i == 3){
                    dst = new Point(areaCenters.get(i).first.getX(), goal.first.getY() - 0.2, goal.first.getZ());

                }
                else{
                    dst = new Point(goal.first.getX(), goal.first.getY(), areaCenters.get(i).first.getZ());
                }

                System.out.println("Area " + i +" Next Coordinate: " +dst.toString());
                moveToWithCheck(dst, areaCenters.get(i).second, false);
            } else {
                System.out.println("❌ AR tag pose not computed — skipping movement.");
            }

            // Pre-processing
            image = api.getMatNavCam();
            image = undistortedImage(image);
            api.saveMatImage(image, "undistorted_area_" + id + ".png");
            double top = cropParams[i][0];
            double bottom = cropParams[i][1];
            double left = cropParams[i][2];
            double right = cropParams[i][3];
            // cropping
            image = cropArea(image, top, bottom, left, right);

            // detecting the items
            analyzeAndStoreAreaItems(image, id, foundItemsPerArea);

            // save a cropped image for debugging
            areaImages.add(image);
            api.saveMatImage(image, "cropped_area" + (i + 1) + ".png");
            System.out.println("Astrobee Quaternion: " + api.getRobotKinematics().getOrientation().toString());

        }

        // complete exploring 4 areas
        api.reportRoundingCompletion();

//        image = api.getMatNavCam();
//        Pair<Point, Quaternion> goal = computeTagApproachPose(image);
//        if (goal != null) {
//            Point dst = new Point(goal.first.getX(), goal.first.getY(), areaCenters.get(i).first.getZ());
//            api.moveTo(dst, areaCenters.get(i).second, false);
//        } else {
//            System.out.println("❌ AR tag pose not computed — skipping movement.");
//        }

        /* ******************************************************************************** */
//        try {
//            Thread.sleep(2000); // 1000 ms = 1 second
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        //------------------------------------------------- Treasure Finding ---------------------------------------------------\
        // Move to the Astronaut
        Point point = new Point(11.243d, -6.7607d, 4.9654d);
        Quaternion quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        moveToWithCheck(point, quaternion, false);

        // detect the treasure
        image = api.getMatNavCam();
        api.saveMatImage(image, "target_area.png");
        image = undistortedImage(image);
        image = cropArea(image, 0.2, 0.2, 0.1, 0.4);
        api.saveMatImage(image, "cropped_target_area.png");
        String treasure = findTheTreasure(image);

        // find the area that contains the treasure item
        int treasureArea = -1;
        for (int i = 0; i < foundItemsPerArea.size(); i++) {
            Map<String, Integer> areaMap = foundItemsPerArea.get(i);

            for (Map.Entry<String, Integer> entry : areaMap.entrySet()) {
                System.out.println("  Item: " + entry.getKey() + " → Count: " + entry.getValue());
                if ( entry.getKey() == treasure){
                    treasureArea = i;
                    break;
                }
            }
            if(treasureArea != -1){
                break;
            }
        }

        api.notifyRecognitionItem();

        // --------------- re-check the found items ---------------------
        for (int i = 0; i < foundItemsPerArea.size(); i++) {
            Map<String, Integer> areaMap = foundItemsPerArea.get(i);
            System.out.println("Area " + i + ":");
            for (Map.Entry<String, Integer> entry : areaMap.entrySet()) {
                System.out.println("  Item: " + entry.getKey() + " → Count: " + entry.getValue());
            }
        }



        if(treasureArea != -1){
            moveToWithCheck(areaCenters.get(treasureArea).first, areaCenters.get(treasureArea).second, false);
        }else {
            System.out.println("Treasure Area NOT found");
        }

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
    private void moveToWithCheck(Point targetPosition, Quaternion targetOrientation, boolean tmp) {
        final double POSITION_TOLERANCE = 0.05; // meters
        final double ORIENTATION_TOLERANCE = 0.1; // cosine of angle difference
        for(int i = 0 ; i < 3 ; i++){
            api.moveTo(targetPosition, targetOrientation, false);

            // Get current robot state
            Kinematics currentKinematics = api.getRobotKinematics();
            Point currentPos = currentKinematics.getPosition();
            Quaternion currentQuat = currentKinematics.getOrientation();

            // Check position difference
            double dx = currentPos.getX() - targetPosition.getX();
            double dy = currentPos.getY() - targetPosition.getY();
            double dz = currentPos.getZ() - targetPosition.getZ();
            double positionError = Math.sqrt(dx*dx + dy*dy + dz*dz);

            // Check orientation similarity (cosine of half-angle)
            double dotProduct = currentQuat.getX() * targetOrientation.getX() +
                    currentQuat.getY() * targetOrientation.getY() +
                    currentQuat.getZ() * targetOrientation.getZ() +
                    currentQuat.getW() * targetOrientation.getW();

            boolean positionOk = positionError < POSITION_TOLERANCE;
            boolean orientationOk = Math.abs(dotProduct) > (1.0 - ORIENTATION_TOLERANCE);

            if (positionOk && orientationOk) {
                System.out.println("Reached target.");
                break;
            }

        }
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
                    api.saveMatImage(croppedRegion, "contour_" + j + ".png");
                    significantRegions.add(croppedRegion);
                }
            }
        }

        return significantRegions;
    }

    private boolean isClosedContour(MatOfPoint cnt) {
        org.opencv.core.Point[] pts = cnt.toArray();
        if (pts.length < 2) return false;
        double distance = Math.sqrt(Math.pow(pts[0].x - pts[pts.length - 1].x, 2) + Math.pow(pts[0].y - pts[pts.length - 1].y, 2));
        return distance < 20;
    }

    // Utility method
    private Mat cropArea(Mat input, double topPercent, double bottomPercent,
                         double leftPercent, double rightPercent) {
        int width = input.cols();
        int height = input.rows();

        // Calculate margins
        int marginTop = (int)(height * topPercent);
        int marginBottom = (int)(height * bottomPercent);
        int marginLeft = (int)(width * leftPercent);
        int marginRight = (int)(width * rightPercent);

        // Calculate crop width and height
        int cropWidth = width - marginLeft - marginRight;
        int cropHeight = height - marginTop - marginBottom;

        // Define ROI and return cropped image
        Rect roi = new Rect(marginLeft, marginTop, cropWidth, cropHeight);
        return new Mat(input, roi);
    }

    public Pair<Point, Quaternion> computeTagApproachPose(Mat input_image) {
        // Step 0: Resize image to match calibration
        Mat resized = new Mat();
        Imgproc.resize(input_image, resized, new Size(1280, 960));

        // Step 1: Detect ArUco marker
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        DetectorParameters parameters = DetectorParameters.create();
        Aruco.detectMarkers(resized, dictionary, corners, ids, parameters);

        if (ids.empty() || corners.isEmpty()) {
            System.out.println("❌ No AR tag detected or corners missing.");
            return null;
        }

        // Step 2: Estimate pose
        Mat rvecs = new Mat();
        Mat tvecs = new Mat();
        float markerLength = 0.05f;
        Mat K = new Mat(3, 3, CvType.CV_64F);
        K.put(0, 0,
                523.105750, 0.000000, 635.434258,
                0.000000, 534.765913, 500.335102,
                0.000000, 0.000000, 1.000000
        );
        MatOfDouble distCoeffs = new MatOfDouble(
                -0.164787, 0.020375, -0.001572, -0.000369, 0.0
        );
        Aruco.estimatePoseSingleMarkers(corners, markerLength, K, distCoeffs, rvecs, tvecs);
        if (rvecs.empty() || tvecs.empty()) return null;

        Mat rvec = new Mat(3, 1, CvType.CV_64F);
        Mat tvec = new Mat(3, 1, CvType.CV_64F);
        double[] tRaw = tvecs.get(0, 0);
        double[] rRaw = rvecs.get(0, 0);
        for (int i = 0; i < 3; i++) {
            tvec.put(i, 0, tRaw[i]);
            rvec.put(i, 0, rRaw[i]);
        }

        // Step 3: Convert rvec to rotation matrix
        Mat R_cam_to_tag = new Mat();
        Calib3d.Rodrigues(rvec, R_cam_to_tag);

        // Step 4: Build T_cam_to_tag [4x4]
        Mat T_cam_to_tag = Mat.eye(4, 4, CvType.CV_64F);
        R_cam_to_tag.copyTo(T_cam_to_tag.submat(0, 3, 0, 3));
        for (int i = 0; i < 3; i++) {
            T_cam_to_tag.put(i, 3, tvec.get(i, 0)[0]);
        }

        // Step 5: Get T_world_to_cam from Kinematics
        Kinematics kin = api.getRobotKinematics();
        Point pos = kin.getPosition();
        Quaternion quat = kin.getOrientation();
        Mat R_world_to_cam = quaternionToRotationMatrix(quat);
        Mat T_world_to_cam = Mat.eye(4, 4, CvType.CV_64F);
        R_world_to_cam.copyTo(T_world_to_cam.submat(0, 3, 0, 3));
        T_world_to_cam.put(0, 3, pos.getX());
        T_world_to_cam.put(1, 3, pos.getY());
        T_world_to_cam.put(2, 3, pos.getZ());

        // Step 6: Multiply to get tag pose in world frame
        Mat T_world_to_tag = new Mat();
        Core.gemm(T_world_to_cam, T_cam_to_tag, 1, new Mat(), 0, T_world_to_tag);

        // Step 7: Retreat based on area direction
        int areaId = (int) ids.get(0, 0)[0];
        areaId = areaId % 10;  // e.g., 101 % 10 = 1

        Mat offset = Mat.zeros(4, 1, CvType.CV_64F);
        switch (areaId) {
            case 1: offset.put(1, 0, 0.3); break; // +Y
            case 2:
            case 3:
                offset.put(2, 0, 0.3); break; // +Z
            case 4: offset.put(0, 0, 0.3); break; // +X
        }
        offset.put(3, 0, 1.0);

        Mat tagCameraPose = new Mat();
        Core.gemm(T_world_to_tag, offset, 1, new Mat(), 0, tagCameraPose);

        Point finalPosition = new Point(
                tagCameraPose.get(0, 0)[0],
                tagCameraPose.get(1, 0)[0],
                tagCameraPose.get(2, 0)[0]
        );
        Mat R_tag = T_world_to_tag.submat(0, 3, 0, 3);
        Quaternion finalOrientation = rotationMatrixToQuaternion(R_tag);

        return new Pair<>(finalPosition, finalOrientation);
    }

    private Mat quaternionToRotationMatrix(Quaternion q) {
        float x = q.getX(), y = q.getY(), z = q.getZ(), w = q.getW();
        Mat R = new Mat(3, 3, CvType.CV_64F);
        R.put(0, 0, 1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w);
        R.put(1, 0, 2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w);
        R.put(2, 0, 2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y);
        return R;
    }

    private Quaternion rotationMatrixToQuaternion(Mat R) {
        double m00 = R.get(0,0)[0], m01 = R.get(0,1)[0], m02 = R.get(0,2)[0];
        double m10 = R.get(1,0)[0], m11 = R.get(1,1)[0], m12 = R.get(1,2)[0];
        double m20 = R.get(2,0)[0], m21 = R.get(2,1)[0], m22 = R.get(2,2)[0];

        double tr = m00 + m11 + m22;
        double qw, qx, qy, qz;

        if (tr > 0) {
            double S = Math.sqrt(tr + 1.0) * 2;
            qw = 0.25 * S;
            qx = (m21 - m12) / S;
            qy = (m02 - m20) / S;
            qz = (m10 - m01) / S;
        } else if ((m00 > m11) & (m00 > m22)) {
            double S = Math.sqrt(1.0 + m00 - m11 - m22) * 2;
            qw = (m21 - m12) / S;
            qx = 0.25 * S;
            qy = (m01 + m10) / S;
            qz = (m02 + m20) / S;
        } else if (m11 > m22) {
            double S = Math.sqrt(1.0 + m11 - m00 - m22) * 2;
            qw = (m02 - m20) / S;
            qx = (m01 + m10) / S;
            qy = 0.25 * S;
            qz = (m12 + m21) / S;
        } else {
            double S = Math.sqrt(1.0 + m22 - m00 - m11) * 2;
            qw = (m10 - m01) / S;
            qx = (m02 + m20) / S;
            qy = (m12 + m21) / S;
            qz = 0.25 * S;
        }

        return new Quaternion((float)qx, (float)qy, (float)qz, (float)qw);
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
            int id = (int) ids.get(0, 0)[0];
            System.out.println("Detected AR tag ID: " + id);
            return id;
        }

        System.out.println("AR tag wasn't detected!!!");
        return -1;
    }

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

    private String findTheTreasure(Mat image){
        String itemType;
        List<Mat> contourImages = CroppedContours(image);
        Set<String> treasureItems = new HashSet<>(Arrays.asList("emerald", "diamond", "crystal"));

        for (Mat i : contourImages){
            itemType = classifyImage(i);
            if(treasureItems.contains(itemType)){
                System.out.println("Treasure Item!!: " +  itemType);
                api.saveMatImage(i, itemType + ".png");
                return itemType;
            }
        }
        System.out.println("Treasure Item NOT found !!!");
        return "";
    }

    private void analyzeAndStoreAreaItems(Mat image, int areaId, List<Map<String, Integer>> foundItemsPerArea) {
        areaId = areaId % 10;
        // Detect and classify items
        List<Mat> croppedImages = CroppedContours(image);
        Map<String, Integer> itemCounts = new HashMap<>();

        for (Mat region : croppedImages) {
            String item = classifyImage(region);
            item = item.contains("(") ? item.substring(0, item.indexOf("(")).trim() : item;

            if (!item.equals("unwanted")) {
                itemCounts.put(item, itemCounts.getOrDefault(item, 0) + 1);
            }
        }

        // Store result in list
        foundItemsPerArea.add(itemCounts);

        // Report to API
        Set<String> ignoreItems = new HashSet<>(Arrays.asList("emerald", "diamond", "crystal"));

        for (Map.Entry<String, Integer> entry : itemCounts.entrySet()) {
            if (!ignoreItems.contains(entry.getKey())) {
                api.setAreaInfo(areaId, entry.getKey(), entry.getValue());
            }
        }
        // Debug
        System.out.println("----- ID : " + areaId);
        System.out.println(new JSONObject(itemCounts).toString());
    }

    // Model
    private void loadModel() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("cls_32.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer modelFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        tflite = new Interpreter(modelFile);
    }

    // Map class ID to label
    private String getClassName(int id) {
        String[] classes = {
                "coin", "compass", "coral", "crystal", "diamond",
                "emerald", "fossil", "key", "letter", "shell", "treasure_box", "unwanted"
        };
        return (id >= 0 && id < classes.length) ? classes[id] : "Unknown";
    }




}