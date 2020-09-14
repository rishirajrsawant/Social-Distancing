package com.example.socialdistancing;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import java.util.ArrayList;
import java.util.List;
import static org.opencv.imgproc.Imgproc.line;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    private Mat frame;
    private HOGDescriptor hogDescriptor;
    private MatOfRect foundLocations;
    private MatOfDouble foundWeights;
    private MatOfFloat descriptors;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.enableFpsMeter();
        cameraBridgeViewBase.setMaxFrameSize(320, 240);


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
        hogDescriptor = new HOGDescriptor();
        foundLocations = new MatOfRect();
        foundWeights = new MatOfDouble();
        descriptors = new MatOfFloat();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.gray();
        List<Point> centers = new ArrayList<Point>();



        hogDescriptor.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());


        hogDescriptor.compute(frame, descriptors);

        hogDescriptor.detectMultiScale(frame, foundLocations, foundWeights, 0, new Size(4,4),
                new Size(8, 8), 1.05);

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2RGBA);

        for(Rect rect : foundLocations.toArray()){
            Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 2);
            double ptCenterX = rect.x + 0.5 * rect.width;
            double ptCenterY = rect.y + 0.5 * rect.height;
            Point center = new Point(ptCenterX,ptCenterY);
            Imgproc.circle(frame, center, 4, new Scalar(255,0,0), 2);
            centers.add(center);

            for (int i = 0; i < centers.size(); i++) {

                for (int j = i + 1; j < centers.size(); j++){
                    if (centers.size() >= 2) {
                        double minDistance = 100;
                        double distance = Math.sqrt(Math.pow(centers.get(i).x  - centers.get(j).x, 2) + Math.pow(centers.get(i).y - centers.get(j).y, 2));
                        line(frame, new Point(centers.get(i).x,centers.get(i).y), new Point(centers.get(j).x,centers.get(j).y), new Scalar(255, 0, 0));
                        if (distance < minDistance) {
                            Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0), 2);
                            Imgproc.putText(frame, "Please follow Social Distancing", new Point(5,230), 0, 0.5, new Scalar(0,0,255), 1);
                        }
                    }
                }
            }

        }
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

