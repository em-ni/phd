/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        cerr << endl << "Usage: ./broncho_rgbd path_to_config.ini" << endl;
        return 1;
    }
    // For debugging use
    // std::cout << __FILE__ << " " << __LINE__ << std::endl;

    // Get the config file path
    string configFilePath = argv[1];
    mINI::INIFile file(configFilePath);
    mINI::INIStructure ini;
    file.read(ini);

    // Get the paths to the vocabulary, settings, record and association files
    string vocabularyPath = ini["RUN"].get("vocabulary");
    string settingsPath = ini["RUN"].get("calibration");
    string recordFolderPath = ini["RUN"].get("record");
    string associationPath = ini["RUN"].get("association");
    string logsPath = ini["RUN"].get("logs");

    // Get running options
    bool withPatient = ini["RUN"].get("patient") == "true";
    bool withEncoder = ini["RUN"].get("encoder") == "true";
    bool useViewer = ini["RUN"].get("viewer") == "true";

    // Encoder data
    std::vector<double> encoderData = {};
    if (withEncoder){
        string encoderPath = ini["ENCODER"].get("sim_encoder");
        std::ifstream csvFile(encoderPath);
        if(!csvFile.is_open()){
            std::cerr << "Could not open encoder file: " << encoderPath << std::endl;
        } else {
            std::string line;
            bool isHeader = true;
            while(std::getline(csvFile, line)) {
                if(isHeader) {
                    isHeader = false;
                    continue;
                }
                std::stringstream ss(line);
                std::string cell, lastCell;
                while(std::getline(ss, cell, ',')) {
                    lastCell = cell;
                }
                try {
                    double value = std::stod(lastCell);
                    encoderData.push_back(value);
                } catch(const std::exception &e) {
                    std::cerr << "Invalid number in CSV: " << lastCell << std::endl;
                }
            }
        }
        std::cout << "Encoder data loaded with size: " << encoderData.size() << std::endl;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImages(associationPath, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    cout << "Loaded " << vstrImageFilenamesRGB.size() << " images" << endl;
    cout << "Loaded " << vstrImageFilenamesD.size() << " depthmaps" << endl;
    cout << "Loaded " << vTimestamps.size() << " timestamps" << endl;
    // Check if the record folder exists
    if (recordFolderPath.empty()){
        cerr << "Record folder path is empty" << endl;
        return 1;
    }

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::RGBD, useViewer, 0, "", configFilePath);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(recordFolderPath + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD = cv::imread(recordFolderPath + "/" + vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: " << recordFolderPath << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        if (withPatient && withEncoder){
            if(ni >= encoderData.size()){
                break;
            }
            double encoderValue = encoderData[ni];
            SLAM.TrackRGBD(imRGB, imD, tframe, encoderValue);
        }
        else{
            SLAM.TrackRGBD(imRGB, imD, tframe);
        }
            

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }


    // Save camera trajectory
    // if it doesn't exist, create the logs folder
    cout << endl << "Saving logs" << endl;
    if (mkdir(logsPath.c_str(), 0777) == -1)
    {
        cerr << "Logs folder already exists" << endl;
    }

    time_t now = time(0);
    struct tm *ltm = localtime(&now);
    char timeStr[20];
    strftime(timeStr, sizeof(timeStr), "%Y%m%d-%H%M%S", ltm);
    string timeSuffix(timeStr);
    SLAM.SaveKeyFrameTrajectoryTUM(logsPath + "/KeyFrameTrajectory_" + timeSuffix + ".txt");
    SLAM.SaveTrajectoryTUM(logsPath + "/CameraTrajectory_" + timeSuffix + ".txt");

    // Stop all threads
    SLAM.Shutdown();
    cout << "SLAM shutdown" << endl;

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl << endl;

    // Sleep for a while
    std::this_thread::sleep_for(std::chrono::seconds(2));
    cout << endl << "End of the program" << endl << endl;

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
