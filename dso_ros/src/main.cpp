/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/





#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "util/NumType.h"


#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"

#include "dso_ros/poseMSG.h"
#include "dso_ros/bgr_frameMSG.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>




std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
std::string image_topic = "";
bool useSampleOutput=false;
ros::Publisher curposePub;
ros::Publisher bgrframePub;
ros::Publisher singleframePub;

using namespace dso;

void parseArgument(char* arg)
{
	int option;
	char buf[1000];

	if(1==sscanf(arg,"sampleoutput=%d",&option))
	{
		if(option==1)
		{
			useSampleOutput = true;
			printf("USING SAMPLE OUTPUT WRAPPER!\n");
		}
		return;
	}

	if(1==sscanf(arg,"quiet=%d",&option))
	{
		if(option==1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}


	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignetteFile = buf;
		printf("loading vignette from %s!\n", vignetteFile.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaFile = buf;
		printf("loading gammaCalib from %s!\n", gammaFile.c_str());
		return;
	}

	printf("could not parse argument \"%s\"!!\n", arg);
}




FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
int frameID = 0;
int refnum = 0;
ros::Time interval;

void vidCb(const sensor_msgs::ImageConstPtr img)
{
	//ros::Time start_t = ros::Time::now();
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
	assert(cv_ptr->image.type() == CV_8U);
	assert(cv_ptr->image.channels() == 1);

	cv_bridge::CvImagePtr bgr_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
	assert(bgr_ptr->image.type() == CV_8UC3);
	assert(bgr_ptr->image.channels() == 3);


	if(setting_fullResetRequested)
	{
		std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
		delete fullSystem;
		for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
		fullSystem = new FullSystem();
		fullSystem->linearizeOperation=false;
		fullSystem->outputWrapper = wraps;
	    if(undistorter->photometricUndist != 0)
	    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
		setting_fullResetRequested=false;
	}

	MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
	ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);

	
	cv::Mat img_bgr;
	auto cameraparam = undistorter -> getOriginalParameter();

	cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << cameraparam[0], 0, cameraparam[2], 0, cameraparam[1], cameraparam[3], 0, 0, 1);
	cv::Mat distCoeffs = (cv::Mat_<double>(1,4) << cameraparam[4], cameraparam[5], cameraparam[6], cameraparam[7]);
	cv::Mat newCameraMatrix;
	cv::eigen2cv(undistorter->getK(), newCameraMatrix);
	cv::undistort(bgr_ptr->image, img_bgr, cameraMatrix, distCoeffs, newCameraMatrix);

	sensor_msgs::Image refframe;
	//refframe = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_bgr).toImageMsg());
	//printf("check\n");
	//singleframePub.publish(refframe);

	fullSystem->addActiveFrame(undistImg, frameID, img_bgr);
	frameID++;
	//printf("DONE-----------------------------------------------------------------------------------------------\n");
	//ros::Time start_t2 = ros::Time::now();
	//printf("normaltime:%f----------\n",(start_t2-start_t).toSec());
	//if(fullSystem -> initialized){
	//	printf("---------------------------------------------------------frameID:%d\n",frameID);
	//}
	//if(fullSystem -> haveNewKeyFrame){
	//	printf("---------------------------------------------------------haveNewKeyFrame\n");
	//}

	if(fullSystem -> haveNewKeyFrame && fullSystem -> initialized && !fullSystem-> slidewindows.empty()&& ros::Time::now()-interval > ros::Duration(0.5)){

    	//refframe = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", fullSystem ->slidewindows.front() ->img).toImageMsg());
		//singleframePub.publish(refframe);
		//printf("keyframe----------------------------------------------------------------------------------------\n");
		interval = ros::Time::now();
		int count=0;
		sensor_msgs::Image refframe;
		refframe = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", fullSystem ->slidewindows.front() ->img).toImageMsg());
		//printf("check\n");
		singleframePub.publish(refframe);
		while(!fullSystem-> slidewindows.empty()){
			dso_ros::bgr_frameMSG framemsg;
			framemsg.msg_id = count++;
			framemsg.window_size = fullSystem -> getWindowSize();
			SE3 frame_pose = fullSystem -> slidewindows.front() -> shell -> camToWorld;
			//printf("check1\n");
			for(int i=0; i<4; i++){
				for(int j=0; j<4; j++){
					framemsg.camToWorld[4*i+j] = frame_pose.matrix()(i,j);
				}
			}
			framemsg.intrinsics[0] = newCameraMatrix.at<double>(0,0);
			framemsg.intrinsics[1] = newCameraMatrix.at<double>(1,1);
			framemsg.intrinsics[2] = newCameraMatrix.at<double>(0,2);
			framemsg.intrinsics[3] = newCameraMatrix.at<double>(1,2);
			//printf("check2\n");

			framemsg.Image = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", fullSystem ->slidewindows.front() ->img).toImageMsg());
			bgrframePub.publish(framemsg);	
			fullSystem -> slidewindows.pop();
		}
		printf("publishing slide windos, total number: %d ----------------------------------------------------------------------\n",refnum++);
	}

	dso_ros::poseMSG posemsg;
	SE3 cur_pose = fullSystem -> getCurPose();
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			posemsg.pose[4*i+j] = cur_pose.matrix()(i,j);
		}
	}
	posemsg.frame_id = frameID;
	curposePub.publish(posemsg);
	
	delete undistImg;

	//printf("ketframetime:%f----------\n",(ros::Time::now()-start_t2).toSec());

}


int main( int argc, char** argv )
{
	ROS_INFO("0");
	ros::init(argc, argv, "dso_live");

	ros::NodeHandle nh("~");

	//ROS_INFO("1");
	nh.param<std::string>("image_topic", image_topic, "/cam0/image_raw");
	nh.param<std::string>("calib", calib, "/home/devil/dso_ws/src/dso_ros-master/euroc_cam.txt");

	nh.param<float>("ImmatureDensity", setting_desiredImmatureDensity, 1000);
	nh.param<float>("PointDensity", setting_desiredPointDensity, 1200);
	nh.param<int>("minFrames", setting_minFrames, 5);
	nh.param<int>("maxFrames", setting_maxFrames, 7);
	nh.param<int>("maxOptIterations", setting_maxOptIterations, 4);
	nh.param<int>("minOptIterations", setting_minOptIterations, 1);
	nh.param<bool>("logStuff", setting_logStuff, false);
	nh.param<float>("kfGlobalWeight", setting_kfGlobalWeight, 1.3);
	nh.param<bool>("quietmode", setting_debugout_runquiet, true);

	interval = ros::Time::now();

	ros::Subscriber imgSub = nh.subscribe(image_topic, 1, &vidCb);

	curposePub = nh.advertise<dso_ros::poseMSG>("/dso/currentpose", 1);
	bgrframePub = nh.advertise<dso_ros::bgr_frameMSG>("dso/bgrframe",100);
	singleframePub = nh.advertise<sensor_msgs::Image>("/dso/reframe", 10);
	

	//ROS_INFO("2");


	//for(int i=1; i<argc;i++) parseArgument(argv[i]);

    /*
	setting_desiredImmatureDensity = 1000;
	setting_desiredPointDensity = 1200;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations=4;
	setting_minOptIterations=1;
	setting_logStuff = false;
	setting_kfGlobalWeight = 1.3;
	*/



	printf("MODE WITH CALIBRATION, but without exposure times!\n");
	setting_photometricCalibration = 2;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;



    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=false;


    if(!disableAllDisplay)
	    fullSystem->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
	    		 (int)undistorter->getSize()[0],
	    		 (int)undistorter->getSize()[1]));


    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

    ros::spin();

    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }

    delete undistorter;
    delete fullSystem;

	return 0;
}

