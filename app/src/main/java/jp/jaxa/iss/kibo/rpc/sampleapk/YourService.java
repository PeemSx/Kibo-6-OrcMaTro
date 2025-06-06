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

public class YourService extends KiboRpcService {

    private Interpreter tflite;
    @Override
    protected void runPlan1(){
        api.startMission(); // 🚀 Begin the mission

        try {
            loadModel(); // 🎯 Load TensorFlow Lite model for item classification
        } catch (IOException e) {
            e.printStackTrace();
        }

        // ------------------------ 🗺️ Area Setup ------------------------
        ArrayList<Pair<Point, Quaternion>> areaCenters = new ArrayList<>(Arrays.asList(
                new Pair<>(new Point(11.0, -10.0, 5.25), new Quaternion(0f, 0f, -0.707f, 0.707f)),   // Area 1
                new Pair<>(new Point(10.75, -8.75, 4.4), new Quaternion(0f, 0.707f, 0, 0.707f)),     // Area 2
                new Pair<>(new Point(11.0, -7.6, 4.4), new Quaternion(0f, 0.707f, 0, 0.707f)),       // Area 3
                new Pair<>(new Point(10.6, -6.76, 4.96), new Quaternion(0f, 0f, 1, 0))               // Area 4
        ));

        List<Map<String, Integer>> foundItemsPerArea = new ArrayList<>();
        List<Mat> areaImages = new ArrayList<>();
        double[][] cropParams = {
                {0.0, 0.0, 0.0, 0.1},   // Area 1
                {0.0, 0.0, 0.2, 0.2},   // Area 2
                {0.0, 0.0, 0.15, 0.25}, // Area 3
                {0.0, 0.0, 0.0, 0.0}    // Area 4 (no crop)
        };

        // ------------------------ 🔍 Explore Each Area ------------------------
        for (int i = 0; i < 4; i++) {
            int areaId = 101 + i;

            // Move to area
            moveToWithCheck(areaCenters.get(i).first, areaCenters.get(i).second, false);

            // Capture and save initial image
            Mat image = api.getMatNavCam();
            api.saveMatImage(image, "area_" + areaId + ".png");

            // Use flashlight in Area 103 for better lighting
            if (areaId == 103) api.flashlightControlFront(0.2f);
            image = api.getMatNavCam();
            if (areaId == 103) api.flashlightControlFront(0f);

            // Undistort and crop image
            image = undistortedImage(image);
            api.saveMatImage(image, "undistorted_area_" + areaId + ".png");

            image = cropArea(image, cropParams[i][0], cropParams[i][1], cropParams[i][2], cropParams[i][3]);
            api.saveMatImage(image, "cropped_area" + (i + 1) + ".png");

            // Analyze cropped image
            analyzeAndStoreAreaItems(image, areaId, foundItemsPerArea);
            areaImages.add(image);

            System.out.println("Astrobee Orientation: " + api.getRobotKinematics().getOrientation());
        }

        // ------------------------ 👨‍🚀 Move to Astronaut ------------------------
        moveToAstronaut();
        api.reportRoundingCompletion();

        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // ------------------------ 💎 Identify the Target Item ------------------------
        Mat targetImg = api.getMatNavCam();
        api.saveMatImage(targetImg, "target_area.png");

        targetImg = undistortedImage(targetImg);
        targetImg = cropArea(targetImg, 0.4, 0.4, 0.4, 0.4);
        api.saveMatImage(targetImg, "cropped_target_area.png");

        String treasure = findTheTreasure(targetImg);
        api.notifyRecognitionItem();

        // ------------------------ 🔎 Match Item to Area ------------------------
        int treasureArea = -1;
        for (int i = 0; i < foundItemsPerArea.size(); i++) {
            Map<String, Integer> areaMap = foundItemsPerArea.get(i);
            if (areaMap.containsKey(treasure)) {
                treasureArea = i;
                break;
            }
        }

        // Print all items detected
        for (int i = 0; i < foundItemsPerArea.size(); i++) {
            System.out.println("Area " + (i + 1) + ":");
            for (Map.Entry<String, Integer> entry : foundItemsPerArea.get(i).entrySet()) {
                System.out.println("  Item: " + entry.getKey() + " → Count: " + entry.getValue());
            }
        }

        // ------------------------ 🎯 Approach Treasure ------------------------
        if (treasureArea != -1) {
            moveToWithCheck(areaCenters.get(treasureArea).first, areaCenters.get(treasureArea).second, false);

            Mat newImage = api.getMatNavCam();
            api.saveMatImage(undistortedImage(newImage), "treasure_area_first.png");

            Pair<Point, Quaternion> goal = computePerpendicularPose(newImage, treasureArea + 1, 0.1);
            if (goal != null) {
                moveToWithCheck(goal.first, areaCenters.get(treasureArea).second, false);
                Mat finalImage = api.getMatNavCam();
                api.saveMatImage(undistortedImage(finalImage), "treasure_area_AR.png");
            } else {
                System.out.println("❌ AR tag pose not computed — skipping movement.");
            }
        } else {
            System.out.println("❌ Treasure Area NOT found");
        }

        // ✅ Finish mission
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

    private void moveToAstronaut(){
        Point point = new Point(11.1d, -6.7d, 4.965d);
        Quaternion quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        moveToWithCheck(point, quaternion, false);
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

    private List<Mat> CroppedContours(Mat inputImage, float tolerance) {
        List<Mat> significantRegions = new ArrayList<>();

        // 2. CLAHE Enhancement
        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        Mat enhanced = new Mat();
        clahe.apply(inputImage, enhanced);

        // 3. Gaussian Blur
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(enhanced, blurred, new Size(5, 5), 0);

        // 4. Canny Edge Detection
        Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, 200, 400);

        // 5. Morphological Closing
        Mat closed = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel);

        // 6. Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(closed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // 7. Filter & crop closed, significant contours
        for (int j = 0; j < contours.size(); j++) {
            MatOfPoint cnt = contours.get(j);
            double area = Imgproc.contourArea(cnt);

            if (area < 100 || !isClosedContour(cnt, tolerance)) continue;

            Rect contourRect = Imgproc.boundingRect(cnt);
            if (contourRect.width < 10 || contourRect.height < 10) continue;

            int cx1 = Math.max(contourRect.x - 5, 0);
            int cy1 = Math.max(contourRect.y - 5, 0);
            int cx2 = Math.min(contourRect.x + contourRect.width + 5, inputImage.width());
            int cy2 = Math.min(contourRect.y + contourRect.height + 5, inputImage.height());

            if (cx2 > cx1 && cy2 > cy1) {
                Mat croppedRegion = new Mat(inputImage, new Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
                api.saveMatImage(croppedRegion, "contour_" + j + ".png");
                significantRegions.add(croppedRegion);
            }
        }

        return significantRegions;
    }

    private boolean isClosedContour(MatOfPoint cnt, float tolerance) {
        org.opencv.core.Point[] pts = cnt.toArray();
        if (pts.length < 2) return false;
        double distance = Math.sqrt(Math.pow(pts[0].x - pts[pts.length - 1].x, 2) + Math.pow(pts[0].y - pts[pts.length - 1].y, 2));
        return distance < tolerance;
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

    private Pair<Point, Quaternion> computePerpendicularPose(Mat rawImage, int areaId, double retreatDist) {
            // Step 1: Resize & undistort
            Mat resized = new Mat();
            Imgproc.resize(rawImage, resized, new Size(1280, 960));

            Mat K = new Mat(3, 3, CvType.CV_64F);
            K.put(0, 0,
                    523.105750, 0.000000, 635.434258,
                    0.000000, 534.765913, 500.335102,
                    0.000000, 0.000000, 1.000000
            );
            MatOfDouble distCoeffs = new MatOfDouble(
                    -0.164787, 0.020375, -0.001572, -0.000369, 0.0
            );

            Mat undistorted = new Mat();
            Calib3d.undistort(resized, undistorted, K, distCoeffs);

            // Step 2: Detect AR tags
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            DetectorParameters parameters = DetectorParameters.create();
            Aruco.detectMarkers(undistorted, dictionary, corners, ids, parameters);

            if (ids.empty()) {
                System.out.println("❌ No AR tag found.");
                return null;
            }

            int targetId = 100 + areaId;
            int tagIndex = -1;
            for (int i = 0; i < ids.rows(); i++) {
                if ((int) ids.get(i, 0)[0] == targetId) {
                    tagIndex = i;
                    break;
                }
            }

            if (tagIndex == -1) {
                System.out.println("❌ AR tag for Area " + areaId + " (ID=" + targetId + ") not found.");
                return null;
            }

            // Step 3: Pose estimation
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            Aruco.estimatePoseSingleMarkers(corners, 0.05f, K, distCoeffs, rvecs, tvecs);

            double[] tvecRaw = tvecs.get(tagIndex, 0);
            Mat tvecCam = new Mat(3, 1, CvType.CV_64F);
            tvecCam.put(0, 0, tvecRaw[0], tvecRaw[1], tvecRaw[2]);

            Kinematics kin = api.getRobotKinematics();
            Point camWorld = kin.getPosition();
            Quaternion camQuat = kin.getOrientation();
            Mat R = quaternionToRotationMatrix(camQuat);

        // Step 3.5: Debug — compute distance and camera angle
        double dx = tvecRaw[0];  // right (+X_cam)
        double dy = tvecRaw[1];  // down (+Y_cam)
        double dz = tvecRaw[2];  // forward (+Z_cam)

        double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);  // Euclidean
        double angleRad = Math.atan2(Math.sqrt(dx * dx + dy * dy), dz);  // in radians
        double angleDeg = Math.toDegrees(angleRad);

        System.out.println("📏 Distance to AR tag: " + String.format("%.3f", distance) + " meters");
        System.out.println("📐 Camera angle to AR tag: " + String.format("%.2f", angleDeg) + "°");


        // Step 4: Transform tag position to world frame
            Mat tagWorldOffset = new Mat();
            Core.gemm(R, tvecCam, 1, new Mat(), 0, tagWorldOffset);
            double tagX = camWorld.getX() + tagWorldOffset.get(0, 0)[0];
            double tagY = camWorld.getY() + tagWorldOffset.get(1, 0)[0];
            double tagZ = camWorld.getZ() + tagWorldOffset.get(2, 0)[0];

            // Step 5: Get camera forward vector (R * Z_cam)
            Mat camForward = new Mat(3, 1, CvType.CV_64F);
            camForward.put(0, 0, 0);
            camForward.put(1, 0, 0);
            camForward.put(2, 0, 1);

            Mat facingDir = new Mat();
            Core.gemm(R, camForward, 1, new Mat(), 0, facingDir);
            double fx = facingDir.get(0, 0)[0];
            double fy = facingDir.get(1, 0)[0];
            double fz = facingDir.get(2, 0)[0];
            double norm = Math.sqrt(fx * fx + fy * fy + fz * fz);
            fx /= norm; fy /= norm; fz /= norm;

            // Step 6: Apply retreat offset
            double goalX = tagX - retreatDist * fx;
            double goalY = tagY - retreatDist * fy;
            double goalZ = tagZ - retreatDist * fz;

            // Step 7: Apply NavCam-to-body offset correction (transform from body to world)
            Mat camOffset = new Mat(3, 1, CvType.CV_64F);
            camOffset.put(0, 0, 0.1177);  // Z_cam = forward
            camOffset.put(1, 0, -0.0422); // Y_cam = downward
            camOffset.put(2, 0, -0.0826); // X_cam = left

            Mat offsetWorld = new Mat();
            Core.gemm(R, camOffset, 1, new Mat(), 0, offsetWorld);

            goalX += offsetWorld.get(0, 0)[0];
            goalY += offsetWorld.get(1, 0)[0];
            goalZ += offsetWorld.get(2, 0)[0];

            // Step 8: Clamp within KIZ
            goalX = Math.max(10.65, Math.min(11.25, goalX));
            goalY = Math.max(-10.0, Math.min(-6.3, goalY));
            goalZ = Math.max(4.4, Math.min(5.3, goalZ));

            Point goalPos = new Point(goalX, goalY, goalZ);
            return new Pair<>(goalPos, camQuat);  // Maintain facing direction
        }

    private Mat quaternionToRotationMatrix(Quaternion q) {
        float x = q.getX(), y = q.getY(), z = q.getZ(), w = q.getW();
        Mat R = new Mat(3, 3, CvType.CV_64F);
        R.put(0, 0, 1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w);
        R.put(1, 0, 2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w);
        R.put(2, 0, 2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y);
        return R;
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
        List<Mat> contourImages = CroppedContours(image, 40);
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
        List<Mat> croppedImages = CroppedContours(image, 13);
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
        AssetFileDescriptor fileDescriptor = getAssets().openFd("new_CLS.tflite");
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