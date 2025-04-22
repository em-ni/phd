#ifndef SKELETON_H
#define SKELETON_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <cassert>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "Thirdparty/Sophus/sophus/sim3.hpp"

#include "Atlas.h"
#include "Map.h"

using namespace std;
namespace ORB_SLAM3 {

class Skeleton {
public:
    Skeleton(string &referenceCenterlinePath, Atlas* pAtlas);
    ~Skeleton();

    void Run();

    void SetCurvilinearAbscissa(double value);
    double GetCurvilinearAbscissa();
    void RequestStop();
    bool IsStopped();
    void SetStopped();

private:
    
    bool mStopRequested = false;
    bool isStopped = false;
    std::mutex mMutexStopRequested;
    std::mutex mMutexStopped;   

    const string serverIP = "127.0.0.1";
    const int serverPort = 12345;
    int serverSocket;
    bool connected = false;
    int InitServer();
    void SendPose();

    Atlas* mpAtlas;

    string mReferenceCenterlinePath;
    std::vector<std::vector<Sophus::SE3f>> mRefCenterlinePoses;
    std::mutex mMutexRefCenterlinePoses;

    void SetReferenceCenterline();
    std::vector<std::vector<Sophus::SE3f>> GetReferenceCenterline();
    std::vector<std::vector<Sophus::SE3f>> FindCandidateTrajectories();

    // Helper function: Resample a trajectory (vector of SE3 poses) to a fixed number of points 
    // along its cumulative arc-length. Returns a vector of 3D positions (Eigen::Vector3d).
    std::vector<Eigen::Vector3d> ResampleTrajectory(const std::vector<Sophus::SE3f>& trajectory, size_t num_samples);

    // Align two trajectories (each given as a vector of SE3 poses)
    // using a Umeyama-like least-squares formulation.
    // The alignment model is: model = s * R * data + t
    // If yaw_only is true, the rotation is restricted to a rotation about Z.
    // known_scale forces s = 1.
    // This version first resamples both trajectories to a fixed number of points
    // so that they have corresponding points even if their original sizes differ.
    Sophus::Sim3f AlignTrajectories(const std::vector<Sophus::SE3f>& model,
        const std::vector<Sophus::SE3f>& data,
        bool known_scale, bool yaw_only);

    // Compute the Absolute Trajectory Error (ATE) between two trajectories after aligning
    // them with a similarity transformation (sim3). To be robust for trajectories of different size,
    // we resample both trajectories to a fixed number of points and then compute the RMSE.
    double CalculateATE(const std::vector<Sophus::SE3f>& model,
                        const std::vector<Sophus::SE3f>& data,
                        const Sophus::Sim3f& sim3);


    // Helper function: Compute the best yaw angle (rotation about Z)
    // that maximizes trace(Rz(theta)*C) following the rpg_trajectory_evaluation approach.
    double GetBestYaw(const Eigen::Matrix3d& C);

    // Helper function: Create a 3x3 rotation matrix for a rotation about the Z-axis.
    Eigen::Matrix3d RotZ(double theta);
    

    // Coming from encoder value 
    double mCurvilinearAbscissa; 
    std::mutex mMutexCurvilinearAbscissa;

    // Current pose on the real lungs wTc
    Sophus::SE3f mCurPose;
    void SetCurPose(Sophus::SE3f pose);
    Sophus::SE3f GetCurPose();
    std::mutex mMutexCurPose;
};

} // namespace ORB_SLAM3

#endif // SKELETON_H