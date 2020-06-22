package com.example.socialdistancing;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.moments;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    private Mat mRgb;
    private Mat mFGMask;
    private Mat frame, blur, thresh, dilate;
    private BackgroundSubtractorMOG2 mog2;
    private Mat hierarchy, tempMat;
    private HOGDescriptor hogDescriptor;
    private MatOfRect foundLocations;
    private MatOfDouble foundWeights;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.enableFpsMeter();
        cameraBridgeViewBase.setMaxFrameSize(640, 480);



        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }

        };

    }




    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgb = new Mat();
        mFGMask = new Mat();
        mog2 = Video.createBackgroundSubtractorMOG2(0, 10, true);
        hierarchy = new Mat();
        blur = new Mat();
        thresh = new Mat();
        dilate = new Mat();
        hogDescriptor = new HOGDescriptor();
        foundLocations = new MatOfRect();
        foundWeights = new MatOfDouble();
        tempMat = new Mat(frame.rows(), frame.cols(), CvType.CV_8UC3);
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.rgba();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();





        // detector = SimpleBlobDetector.create();
        hogDescriptor.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());
        Imgproc.cvtColor(frame, mRgb, Imgproc.COLOR_RGBA2GRAY);
        mog2.apply(frame, mFGMask, 0.4); //apply() exports a gray image by definition
        Imgproc.cvtColor(mFGMask, frame, Imgproc.COLOR_GRAY2RGBA);

//        detector.detect(mFGMask, keypoints, frame);
//        int flags = features2d.DRAW_RICH_KEYPOINTS;
//        features2d.drawKeypoints(mFGMask, keypoints, frame, new Scalar(0,0,255), flags);

//        detector.compute(frame, keypoints, descriptors);



        Imgproc.GaussianBlur(mFGMask, blur, new Size(5, 5), 0);
//      Imgproc.Canny(blur, blur, 80, 100);
        Imgproc.threshold(blur, thresh, 20, 255, THRESH_BINARY);
        Imgproc.dilate(thresh, dilate, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));
        Imgproc.findContours(dilate, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        Imgproc.cvtColor(frame, tempMat, Imgproc.COLOR_RGBA2RGB);

        for ( int contourIdx=0; contourIdx < contours.size(); contourIdx++ )
        {
            hogDescriptor.detectMultiScale(tempMat, foundLocations, foundWeights, 1.5, new Size(8,8),
                    new Size(32, 32), 1.05);
            // Minimum size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(contourIdx).toArray() );
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );



            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(points);

            //double contourArea = Imgproc.contourArea(contours.get(contourIdx));









            //  if (contourArea < 100){}
            //else {

//                List<Moments> mu = new ArrayList<Moments>(contours.size());
//                for (int i = 0; i < contours.size(); i++) {
//                    mu.add(i, Imgproc.moments(contours.get(i), false));
//                    Moments p = mu.get(i);
//                    int x = (int) (p.get_m10() / p.get_m00());
//                    int y = (int) (p.get_m01() / p.get_m00());
//                    Imgproc.circle(frame, new Point(x, y), 4, new Scalar(0, 255, 0));
//                }
            //Log.i("area798", "a:"+contourArea);
            Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 3);
            // }

        }




//        Imgproc.drawContours(mRgb, contours, -1, new Scalar(0, 255, 0), 2);

//        Imgproc.erode(mFGMask, frame, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
//        Imgproc.dilate(mFGMask, frame, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));
//        Imgproc.morphologyEx(mFGMask, frame, MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));
//        Imgproc.morphologyEx(mFGMask, frame, MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));


//        //System.gc();
        contours.clear();
        return frame;
    }


    @Override
    public void onCameraViewStopped() {
        frame.release();
    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"OpenCV not loaded properly!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}

